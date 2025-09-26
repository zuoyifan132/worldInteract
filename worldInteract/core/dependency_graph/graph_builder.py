"""
Dependency Graph Builder for modeling tool relationships and domain clustering.
"""

import json
import networkx as nx
import numpy as np
import textwrap
from typing import Dict, List, Any, Tuple, Optional, Set
from pathlib import Path
from loguru import logger
from community import community_louvain
import matplotlib.pyplot as plt
from tqdm import tqdm

from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.embedding import OpenAIEmbeddings
from worldInteract.utils.parser_utils import extract_json_from_text


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
        self.handle_singleton = self.env_config.get("handle_singleton", False)
        self.singleton_tool_similarity_threshold = self.env_config.get("singleton_tool_similarity_threshold", 0.6)
        
        logger.info(f"Initialized Dependency Graph Builder with threshold: {self.similarity_threshold}")
        logger.info(f"Singleton handling: {'enabled' if self.handle_singleton else 'disabled'}")
    
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
        communities, singleton_tools = self._detect_communities(graph)
        
        # Step 4: Validate communities and create domains with LLM (if enabled)
        # TODO: return outlier tools append to singleton tools
        if self.enable_llm_validation:
            logger.info("Validating communities and creating domains with LLM...")
            domains = self._validate_communities_with_llm(apis, communities)
        else:
            raise ValueError("LLM validation is not enabled for graph building, please enable it in the config")
        
        # Step 6: Handle singleton tools (if enabled)
        if singleton_tools and self.handle_singleton:
            logger.info(f"Handling {len(singleton_tools)} singleton tools...")
            domains = self._handle_singleton_tools(singleton_tools, domains, apis)
        elif singleton_tools:
            logger.info(f"Skipping handling of {len(singleton_tools)} singleton tools (disabled in config)")
        
        # Step 7: Save outputs
        output_data = self._save_outputs(
            apis, graph, communities, domains, tool_embeddings, output_dir
        )
        
        logger.info(f"Created {len(domains)} domains with {len(apis)} total tools")
        return output_data
    
    def _generate_tool_embeddings(self, apis: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
        """Generate embeddings for all tool parameters."""
        tool_embeddings = {}
        
        for api in tqdm(apis, desc="Generating tool embeddings"):
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
    
    def _detect_communities(self, graph: nx.Graph) -> Tuple[Dict[int, List[str]], List[str]]:
        """Detect communities using Louvain algorithm.
        
        Returns:
            Tuple of (communities, singleton_tools)
        """
        if graph.number_of_edges() == 0:
            # No edges, each node is its own singleton tool
            singleton_tools = list(graph.nodes())
            communities = {}
            logger.warning("No edges found, all tools are singleton tools")
            return communities, singleton_tools
        
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
                    logger.warning(f"Community {comm_id} is too large, the size is {len(tools)}, add it for now")
                    # TODO: Handle to large communities 
                    filtered_communities[len(filtered_communities)] = tools
            else:
                singleton_tools.extend(tools)
        
        logger.info(f"Detected {len(filtered_communities)} communities and {len(singleton_tools)} singleton tools")
        for comm_id, tools in filtered_communities.items():
            logger.debug(f"Community {comm_id}: {len(tools)} tools - {', '.join(tools[:3])}{'...' if len(tools) > 3 else ''}")
        
        if singleton_tools:
            logger.debug(f"Singleton tools: {', '.join(singleton_tools[:5])}{'...' if len(singleton_tools) > 5 else ''}")
        
        return filtered_communities, singleton_tools
    
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
    
    # Note: _merge_singleton_tools method removed as singleton tools are now handled 
    # in _handle_singleton_tools after domain creation
    
    def _handle_singleton_tools(
        self,
        singleton_tools: List[str],
        domains: List[Dict[str, Any]],
        apis: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Handle singleton tools by comparing with existing domains and using LLM validation.
        
        Args:
            singleton_tools: List of tool names that don't belong to any community
            domains: List of existing domain objects
            apis: List of all API objects
            
        Returns:
            Updated list of domains with singleton tools added where appropriate
        """
        api_dict = {api["name"]: api for api in apis}
        unassigned_tools = []
        
        for tool_name in tqdm(singleton_tools, desc="Handling singleton tools"):
            if tool_name not in api_dict:
                logger.warning(f"Singleton tool {tool_name} not found in APIs")
                continue
                
            tool_api = api_dict[tool_name]
            best_domain_idx = None
            best_similarity = 0.0
            
            # Calculate similarity with each domain using descriptions
            for domain_idx, domain in enumerate(domains):
                try:
                    # Calculate similarity between tool description and domain description
                    tool_description = tool_api["description"]
                    domain_description = domain["description"]
                    
                    # Get embeddings for descriptions
                    descriptions = [tool_description, domain_description]
                    embeddings = self.embeddings.embed_texts(descriptions)
                    
                    if len(embeddings) == 2:
                        tool_desc_embedding = embeddings[0]
                        domain_desc_embedding = embeddings[1]
                        
                        # Calculate cosine similarity
                        similarity = self.embeddings.cosine_similarity(tool_desc_embedding, domain_desc_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_domain_idx = domain_idx
                        
                except Exception as e:
                    logger.error(f"Failed to calculate similarity between {tool_name} description and domain '{domain['domain']}' description: {e}")
            
            # Check if similarity exceeds threshold
            if best_similarity > self.singleton_tool_similarity_threshold and best_domain_idx is not None:
                # Use LLM to validate if the tool should join this domain
                domain = domains[best_domain_idx]
                should_join = self._llm_validate_singleton_tool(tool_api, domain)
                
                if should_join:
                    # Add tool to the domain
                    domains[best_domain_idx]["tools"].append(tool_api)
                    domains[best_domain_idx]["tool_count"] += 1
                    logger.info(f"Added singleton tool {tool_name} to domain '{domain['domain']}' (similarity: {best_similarity:.3f})")
                else:
                    unassigned_tools.append(tool_api)
                    logger.info(f"LLM rejected adding {tool_name} to domain '{domain['domain']}' (similarity: {best_similarity:.3f})")
            else:
                unassigned_tools.append(tool_api)
                logger.info(f"Singleton tool {tool_name} similarity too low (best: {best_similarity:.3f}, threshold: {self.singleton_tool_similarity_threshold})")
        
        return domains
    
    def _llm_validate_singleton_tool(self, tool_api: Dict[str, Any], domain: Dict[str, Any]) -> bool:
        """
        Use LLM to validate if a singleton tool should join a domain.
        
        Args:
            tool_api: The singleton tool API object
            domain: The domain object
            
        Returns:
            True if the tool should join the domain, False otherwise
        """
        try:
            system_prompt = textwrap.dedent("""
                You are an API analysis expert. Please determine whether a singleton tool should join an existing functional domain.
                
                Evaluation criteria:
                1. Does the tool's functionality relate to other tools in the domain?
                2. Can the tool work collaboratively with tools in the domain?
                3. Does the tool belong to the same application scenario or business domain?
                4. Would adding it enhance the overall functional consistency of the domain?
                
                Please answer only "Yes" or "No" and provide a brief explanation.
            """).strip()

            user_prompt = textwrap.dedent(f"""
                Please determine whether the following tool should join the specified functional domain:

                Candidate Tool:
                - Name: {tool_api['name']}
                - Description: {tool_api['description']}

                Target Functional Domain:
                - Domain Name: {domain['domain']}
                - Domain Description: {domain['description']}
                - Current Tool Count: {domain['tool_count']}
                - Sample Tools: {', '.join([tool['name'] for tool in domain['tools'][:3]])}

                Should this tool join the functional domain?
            """).strip()

            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=300
            )
            
            # Parse response
            response = answer_text.strip().lower()
            should_join = "yes" in response
            
            return should_join
            
        except Exception as e:
            logger.error(f"LLM validation failed for singleton tool {tool_api['name']}: {e}")
            # Default to not joining if validation fails
            return False
    
    def _validate_communities_with_llm(
        self, 
        apis: List[Dict[str, Any]], 
        communities: Dict[int, List[str]]
    ) -> List[Dict[str, Any]]:
        """Validate communities and create domains using LLM analysis."""
        api_dict = {api["name"]: api for api in apis}
        domains = []
        
        for comm_id, tools in communities.items():
            if len(tools) <= 1:
                # Handle single tools separately
                if tools and tools[0] in api_dict:
                    tool_api = api_dict[tools[0]]
                    domain = {
                        "domain": f"{tool_api['name']}_operations",
                        "description": f"Single tool domain: {tool_api['description'][:50]}...",
                        "tool_count": 1,
                        "tools": [tool_api]
                    }
                    domains.append(domain)
                continue
            
            try:
                # Get tool descriptions for LLM analysis
                tool_descriptions = []
                for tool in tools:
                    if tool in api_dict:
                        tool_descriptions.append(f"- {tool}: {api_dict[tool]['description']}")
                
                # Ask LLM to analyze and reorganize tools into proper domains
                llm_domains = self._llm_analyze_and_create_domains(tool_descriptions, api_dict)
                
                if llm_domains:
                    domains.extend(llm_domains)
                    logger.info(f"LLM analysis created {len(llm_domains)} domains for community {comm_id}")
                else:
                    logger.warning(f"LLM analysis failed for community {comm_id}, skip this community for tool names {tools}")
                    continue
                
            except Exception as e:
                logger.error(f"LLM validation failed for community {comm_id}: {e}, skip this community for tool names {tools}")
                continue
        
        logger.info(f"Created {len(domains)} domains with LLM validation")
        return domains
    
    def _llm_analyze_and_create_domains(
        self, 
        tool_descriptions: List[str], 
        api_dict: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use LLM to analyze tools and create proper domain groupings."""
        
        system_prompt = textwrap.dedent("""
            You are an API analysis expert. Please analyze the given tools and create coherent functional domains.
            
            Your task:
            1. Identify one or more functional domains that best represent the core functionalities
            2. Group tools by their functional relationships - tools that work together or belong to the same business area
            3. Include only tools that clearly belong to each functional domain
            4. Remove tools that don't fit well with any identified domain (outliers)
            5. Generate meaningful domain names and descriptions for each identified domain
            6. Remove functionally duplicate tools within the same domain
            
            Please respond with a JSON structure like this:
            ```json
            {
                "domains": [
                    {
                        "domain_name": "file_operations",
                        "description": "Tools for file system operations and management",
                        "tools": ["create_file", "read_file", "delete_file"]
                    },
                    {
                        "domain_name": "user_management", 
                        "description": "Tools for managing user accounts and authentication",
                        "tools": ["create_user", "authenticate_user", "delete_user"]
                    }
                ]
            }
            ```
            
            Guidelines:
            - Each domain should have at least 2 tools (unless there's only 1 tool total)
            - Tools within a domain should have clear functional relationships
            - It's better to exclude outliers than force them into inappropriate domains
        """).strip()

        tools_text = "\n".join(tool_descriptions)
        user_prompt = textwrap.dedent(f"""
            Please analyze the following tools and identify the most appropriate functional domains:
            
            {tools_text}
            
            Group these tools into coherent functional domains based on their relationships and functionalities. Include only tools that clearly belong to each domain, and exclude any outliers that don't fit well.
        """).strip()

        try:
            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=1000
            )
            
            extracted_json = extract_json_from_text(answer_text.strip())
            response_data = json.loads(extracted_json)
            
            domains = []
            
            if "domains" in response_data:
                domains_info = response_data["domains"]
                
                for domain_info in domains_info:
                    domain_name = domain_info.get("domain_name", "unknown_domain")
                    description = domain_info.get("description", "Generated domain")
                    tool_names = domain_info.get("tools", [])
                    
                    # Collect tool objects
                    domain_tools = []
                    for tool_name in tool_names:
                        if tool_name in api_dict:
                            domain_tools.append(api_dict[tool_name])
                        else:
                            logger.warning(f"Tool {tool_name} not found in api_dict")
                    
                    if domain_tools:  # Only create domain if it has valid tools
                        domain = {
                            "domain": domain_name,
                            "description": description,
                            "tool_count": len(domain_tools),
                            "tools": domain_tools
                        }
                        domains.append(domain)
                        logger.info(f"Created domain '{domain_name}' with {len(domain_tools)} tools")
            
            return domains
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return []
        except Exception as e:
            logger.error(f"LLM domain analysis failed: {e}")
            return []

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
