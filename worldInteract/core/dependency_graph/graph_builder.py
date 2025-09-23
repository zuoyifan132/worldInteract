"""
Dependency Graph Builder for modeling tool relationships and domain clustering.
"""

import json
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from pathlib import Path
from loguru import logger
from community import community_louvain
import matplotlib.pyplot as plt

from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.embedding import OpenAIEmbeddings


class DependencyGraphBuilder:
    """Builds tool dependency graphs and performs domain clustering."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize dependency graph builder.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.model_config = self.config_manager.get_model_config("dependency_graph")
        self.env_config = self.config_manager.get_environment_config("dependency_graph")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Configuration
        self.similarity_threshold = self.env_config.get("similarity_threshold", 0.75)
        self.min_community_size = self.env_config.get("min_community_size", 2)
        self.max_community_size = self.env_config.get("max_community_size", 20)
        self.louvain_resolution = self.env_config.get("louvain_resolution", 1.0)
        self.enable_llm_validation = self.env_config.get("enable_llm_validation", True)
        
        logger.info(f"Initialized Dependency Graph Builder with threshold: {self.similarity_threshold}")
    
    def build_dependency_graph(
        self, 
        cleaned_apis_path: str, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Build tool dependency graph and perform domain clustering.
        
        Args:
            cleaned_apis_path: Path to cleaned APIs JSON file
            output_dir: Directory to save graph and domain outputs
            
        Returns:
            Dictionary with graph statistics and domain assignments
        """
        logger.info(f"Building dependency graph from {cleaned_apis_path}")
        
        # Load cleaned APIs
        with open(cleaned_apis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        apis = data.get("apis", [])
        logger.info(f"Loaded {len(apis)} cleaned APIs")
        
        # Step 1: Generate parameter embeddings for all tools
        logger.info("Generating parameter embeddings...")
        tool_embeddings = self._generate_tool_embeddings(apis)
        
        # Step 2: Build similarity graph
        logger.info("Building similarity graph...")
        graph = self._build_similarity_graph(apis, tool_embeddings)
        
        # Step 3: Perform community detection
        logger.info("Performing community detection...")
        communities = self._detect_communities(graph)
        
        # Step 4: Validate dependencies with LLM (if enabled)
        if self.enable_llm_validation:
            logger.info("Validating dependencies with LLM...")
            communities = self._validate_communities_with_llm(apis, communities)
        
        # Step 5: Create domain assignments
        logger.info("Creating domain assignments...")
        domains = self._create_domains(apis, communities)
        
        # Step 6: Save outputs
        output_data = self._save_outputs(
            apis, graph, communities, domains, tool_embeddings, output_dir
        )
        
        logger.info(f"Created {len(domains)} domains with {len(apis)} total tools")
        return output_data
    
    def _generate_tool_embeddings(self, apis: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
        """Generate embeddings for all tool parameters."""
        tool_embeddings = {}
        
        for api in apis:
            tool_name = api["name"]
            try:
                embeddings = self.embeddings.embed_tool_parameters(api)
                tool_embeddings[tool_name] = embeddings
                logger.debug(f"Generated embeddings for {tool_name}: {len(embeddings)} parameters")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {tool_name}: {e}")
                tool_embeddings[tool_name] = {}
        
        return tool_embeddings
    
    def _build_similarity_graph(
        self, 
        apis: List[Dict[str, Any]], 
        tool_embeddings: Dict[str, Dict[str, List[float]]]
    ) -> nx.Graph:
        """Build graph with edges based on parameter similarity."""
        graph = nx.Graph()
        
        # Add all tools as nodes
        for api in apis:
            graph.add_node(api["name"], **api)
        
        # Calculate similarities and add edges
        tool_names = [api["name"] for api in apis]
        edge_count = 0
        
        # all tools will be compare just one time
        for i, tool1 in enumerate(tool_names):
            for j, tool2 in enumerate(tool_names[i+1:], i+1):
                try:
                    similarity = self.embeddings.calculate_tool_similarity(
                        tool_embeddings.get(tool1, {}),
                        tool_embeddings.get(tool2, {})
                    )
                    
                    if similarity > self.similarity_threshold:
                        graph.add_edge(tool1, tool2, weight=similarity)
                        edge_count += 1
                        logger.debug(f"Added edge: {tool1} <-> {tool2} (similarity: {similarity:.3f})")
                
                except Exception as e:
                    logger.error(f"Failed to calculate similarity between {tool1} and {tool2}: {e}")
        
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {edge_count} edges")
        return graph
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Louvain algorithm."""
        if graph.number_of_edges() == 0:
            # No edges, each node is its own community
            communities = {i: [node] for i, node in enumerate(graph.nodes())}
            logger.warning("No edges found, each tool will be its own domain")
            return communities
        
        # Apply Louvain community detection
        partition = community_louvain.best_partition(graph, resolution=self.louvain_resolution)
        
        # Group tools by community
        communities = {}
        for tool, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(tool)
        
        # Filter communities by size
        filtered_communities = {}
        singleton_tools = []
        
        for comm_id, tools in communities.items():
            if len(tools) >= self.min_community_size:
                if len(tools) <= self.max_community_size:
                    filtered_communities[len(filtered_communities)] = tools
                else:
                    # Split large communities
                    split_communities = self._split_large_community(tools, graph)
                    for split_comm in split_communities:
                        filtered_communities[len(filtered_communities)] = split_comm
            else:
                singleton_tools.extend(tools)
        
        # Handle singleton tools - try to merge with most similar community
        if singleton_tools:
            filtered_communities = self._merge_singleton_tools(
                singleton_tools, filtered_communities, graph
            )
        
        logger.info(f"Detected {len(filtered_communities)} communities")
        for comm_id, tools in filtered_communities.items():
            logger.debug(f"Community {comm_id}: {len(tools)} tools - {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
        
        return filtered_communities
    
    def _split_large_community(self, tools: List[str], graph: nx.Graph) -> List[List[str]]:
        """Split large communities into smaller ones."""
        if len(tools) <= self.max_community_size:
            return [tools]
        
        # Create subgraph for this community
        subgraph = graph.subgraph(tools).copy()
        
        # Apply Louvain again with higher resolution
        partition = community_louvain.best_partition(subgraph, resolution=self.louvain_resolution * 2)
        
        # Group tools by new communities
        sub_communities = {}
        for tool, comm_id in partition.items():
            if comm_id not in sub_communities:
                sub_communities[comm_id] = []
            sub_communities[comm_id].append(tool)
        
        # Return communities of appropriate size
        result = []
        for sub_tools in sub_communities.values():
            if len(sub_tools) >= self.min_community_size:
                if len(sub_tools) <= self.max_community_size:
                    result.append(sub_tools)
                else:
                    # Recursively split if still too large
                    result.extend(self._split_large_community(sub_tools, graph))
            else:
                # Handle small subcommunities separately
                result.append(sub_tools)
        
        return result if result else [tools]
    
    def _merge_singleton_tools(
        self, 
        singleton_tools: List[str], 
        communities: Dict[int, List[str]], 
        graph: nx.Graph
    ) -> Dict[int, List[str]]:
        """Merge singleton tools with most similar communities."""
        for tool in singleton_tools:
            best_community = None
            best_similarity = 0.0
            
            # Find community with highest average similarity
            for comm_id, comm_tools in communities.items():
                similarities = []
                for comm_tool in comm_tools:
                    if graph.has_edge(tool, comm_tool):
                        similarities.append(graph[tool][comm_tool]['weight'])
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_community = comm_id
            
            # Add to best community or create new one
            if best_community is not None and best_similarity > self.similarity_threshold * 0.7:
                communities[best_community].append(tool)
            else:
                # Create new community for singleton
                communities[len(communities)] = [tool]
        
        return communities
    
    def _validate_communities_with_llm(
        self, 
        apis: List[Dict[str, Any]], 
        communities: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """Validate and refine communities using LLM analysis."""
        api_dict = {api["name"]: api for api in apis}
        validated_communities = {}
        
        for comm_id, tools in communities.items():
            if len(tools) <= 2:
                # Skip validation for small communities
                validated_communities[len(validated_communities)] = tools
                continue
            
            try:
                # Get tool descriptions for LLM analysis
                tool_descriptions = []
                for tool in tools:
                    if tool in api_dict:
                        tool_descriptions.append(f"- {tool}: {api_dict[tool]['description']}")
                
                # Ask LLM to validate the grouping
                # TODO: the suggestion include a dict which contains domain name and its corresponding 
                # tools belong to it, some outside tools should be removed from the community
                is_valid, suggestions = self._llm_validate_community(tool_descriptions)
                
                if is_valid:
                    validated_communities[len(validated_communities)] = tools
                else:
                    # Split community based on LLM suggestions
                    # TODO: consider remove this functionality since suggestion already split the tools
                    # and remove outlier tools from the community
                    split_communities = self._llm_split_community(tools, suggestions, api_dict)
                    for split_comm in split_communities:
                        validated_communities[len(validated_communities)] = split_comm
                
            except Exception as e:
                logger.error(f"LLM validation failed for community {comm_id}: {e}")
                # Keep original community if validation fails
                validated_communities[len(validated_communities)] = tools
        
        return validated_communities
    
    def _llm_validate_community(self, tool_descriptions: List[str]) -> Tuple[bool, str]:
        """Use LLM to validate if tools belong to the same domain."""
        system_prompt = """你是一个API分析专家。请分析以下工具是否属于同一个功能域。

评判标准：
1. 工具是否解决相似的问题或属于同一个应用场景
2. 工具之间是否有明显的功能关联性
3. 是否可以归类到同一个领域（如文件操作、用户管理、数据库操作等）

请回答"是"或"否"，并简要说明原因。"""

        tools_text = "\n".join(tool_descriptions)
        user_prompt = f"""请分析以下工具组是否属于同一个功能域：

{tools_text}

这些工具是否应该归为同一个域？请回答"是"或"否"，并说明原因。"""

        try:
            response = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=500
            )
            
            # Parse response
            response = response.strip().lower()
            is_valid = "是" in response[:10] or "yes" in response[:10]
            
            return is_valid, response
            
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return True, "Validation failed"
    
    def _llm_split_community(
        self, 
        tools: List[str], 
        llm_suggestions: str, 
        api_dict: Dict[str, Dict[str, Any]]
    ) -> List[List[str]]:
        """Split community based on LLM suggestions."""
        # For now, implement a simple split strategy
        # In practice, you could parse LLM suggestions more sophisticated
        
        if len(tools) <= 3:
            return [tools]
        
        # Simple strategy: split in half
        mid = len(tools) // 2
        return [tools[:mid], tools[mid:]]
    
    def _create_domains(
        self, 
        apis: List[Dict[str, Any]], 
        communities: Dict[int, List[str]]
    ) -> List[Dict[str, Any]]:
        """Create domain objects from communities."""
        api_dict = {api["name"]: api for api in apis}
        domains = []
        
        for comm_id, tools in communities.items():
            # Generate domain name and description
            domain_name = self._generate_domain_name(tools, api_dict)
            domain_description = self._generate_domain_description(tools, api_dict)
            
            # Collect tools for this domain
            domain_tools = []
            for tool_name in tools:
                if tool_name in api_dict:
                    domain_tools.append(api_dict[tool_name])
            
            domain = {
                "domain": domain_name,
                "description": domain_description,
                "tool_count": len(domain_tools),
                "tools": domain_tools
            }
            
            domains.append(domain)
        
        return domains
    
    def _generate_domain_name(self, tools: List[str], api_dict: Dict[str, Dict[str, Any]]) -> str:
        """Generate a descriptive name for the domain."""
        # Simple heuristic: find common words in tool names
        common_words = []
        
        # Extract words from tool names
        all_words = []
        for tool in tools:
            words = tool.replace('_', ' ').split()
            all_words.extend(words)
        
        # Find most common words
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Use most common word or combination
        if sorted_words:
            primary_word = sorted_words[0][0]
            if len(sorted_words) > 1:
                secondary_word = sorted_words[1][0]
                return f"{primary_word}_{secondary_word}"
            else:
                return f"{primary_word}_operations"
        
        # Fallback
        return f"domain_{len(tools)}_tools"
    
    def _generate_domain_description(self, tools: List[str], api_dict: Dict[str, Dict[str, Any]]) -> str:
        """Generate a description for the domain."""
        try:
            # Collect tool descriptions
            descriptions = []
            for tool in tools[:5]:  # Limit to first 5 tools
                if tool in api_dict:
                    descriptions.append(f"- {tool}: {api_dict[tool]['description']}")
            
            system_prompt = """你是一个API分析专家。根据提供的工具列表，生成一个简洁的域描述。

要求：
1. 描述应该概括这些工具的共同功能领域
2. 长度在20-60字之间
3. 使用中文
4. 突出核心功能特征"""

            tools_text = "\n".join(descriptions)
            user_prompt = f"""以下工具属于同一个功能域：

{tools_text}

请为这个功能域生成一个简洁的描述："""

            description = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=200
            )
            
            return description.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate domain description: {e}")
            return f"包含{len(tools)}个相关工具的功能域"
    
    def _save_outputs(
        self,
        apis: List[Dict[str, Any]],
        graph: nx.Graph,
        communities: Dict[int, List[str]],
        domains: List[Dict[str, Any]],
        tool_embeddings: Dict[str, Dict[str, List[float]]],
        output_dir: str
    ) -> Dict[str, Any]:
        """Save all outputs to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save graph data
        graph_data = {
            "nodes": list(graph.nodes(data=True)),
            "edges": [(u, v, data) for u, v, data in graph.edges(data=True)],
            "statistics": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
                "avg_clustering": nx.average_clustering(graph) if graph.number_of_edges() > 0 else 0
            }
        }
        
        with open(output_path / "dependency_graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        # Save communities
        communities_data = {
            "communities": communities,
            "statistics": {
                "community_count": len(communities),
                "avg_community_size": np.mean([len(tools) for tools in communities.values()]),
                "min_community_size": min([len(tools) for tools in communities.values()]) if communities else 0,
                "max_community_size": max([len(tools) for tools in communities.values()]) if communities else 0
            }
        }
        
        with open(output_path / "communities.json", 'w', encoding='utf-8') as f:
            json.dump(communities_data, f, indent=2, ensure_ascii=False)
        
        # Save domains
        domains_data = {
            "metadata": {
                "total_domains": len(domains),
                "total_tools": sum(domain["tool_count"] for domain in domains),
                "avg_tools_per_domain": np.mean([domain["tool_count"] for domain in domains]) if domains else 0
            },
            "domains": domains
        }
        
        with open(output_path / "domains.json", 'w', encoding='utf-8') as f:
            json.dump(domains_data, f, indent=2, ensure_ascii=False)
        
        # Save individual domain files
        domains_dir = output_path / "domains"
        domains_dir.mkdir(exist_ok=True)
        
        for i, domain in enumerate(domains):
            domain_file = domains_dir / f"{domain['domain']}.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(domain, f, indent=2, ensure_ascii=False)
        
        # Save embeddings (optional, for analysis)
        embeddings_data = {
            "tool_embeddings": tool_embeddings,
            "embedding_config": {
                "model": self.embeddings.model,
                "dimensions": self.embeddings.dimensions
            }
        }
        
        with open(output_path / "embeddings.json", 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        # Generate visualization (optional)
        try:
            self._visualize_graph(graph, communities, output_path / "graph_visualization.png")
        except Exception as e:
            logger.warning(f"Failed to generate graph visualization: {e}")
        
        logger.info(f"Saved all outputs to {output_dir}")
        
        return {
            "graph_statistics": graph_data["statistics"],
            "community_statistics": communities_data["statistics"],
            "domain_statistics": domains_data["metadata"],
            "output_directory": str(output_path)
        }
    
    def _visualize_graph(self, graph: nx.Graph, communities: Dict[int, List[str]], output_path: str):
        """Generate graph visualization."""
        if graph.number_of_nodes() == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create color map for communities
        colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
        node_colors = {}
        
        for comm_id, tools in communities.items():
            color = colors[comm_id % len(colors)]
            for tool in tools:
                node_colors[tool] = color
        
        # Set node colors
        node_color_list = [node_colors.get(node, 'gray') for node in graph.nodes()]
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw graph
        nx.draw(
            graph, pos,
            node_color=node_color_list,
            node_size=300,
            font_size=8,
            font_weight='bold',
            edge_color='gray',
            alpha=0.7,
            with_labels=True
        )
        
        plt.title(f"Tool Dependency Graph\n{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {len(communities)} communities")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved graph visualization to {output_path}")
