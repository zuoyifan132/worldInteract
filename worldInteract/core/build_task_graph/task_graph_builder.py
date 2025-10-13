"""
Task Graph Builder for modeling function call dependencies based on parameter similarity.
"""

import json
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt

from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.embedding import OpenAIEmbeddings


class TaskGraphBuilder:
    """Builds task graphs from generated environments based on parameter dependencies."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize task graph builder.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.env_config = self.config_manager.get_environment_config("task_generation")
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Configuration
        self.similarity_threshold = self.env_config.get("parameter_similarity_threshold", 0.7)
        
        logger.info(f"Initialized Task Graph Builder with parameter similarity threshold: {self.similarity_threshold}")
    
    def build_task_graph(
        self,
        generated_env_dirs: List[str],
        domain_graph_dir: str,
        output_dir: str,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build task graph from generated environments.
        
        Args:
            generated_env_dirs: List of paths to generated environment directories
            domain_graph_dir: Path to domain graph directory (contains domain definitions)
            output_dir: Directory to save task graph outputs
            graph_name: Optional name for the task graph
            
        Returns:
            Dictionary with graph statistics and metadata
        """
        logger.info("=" * 80)
        logger.info(f"Building task graph from {len(generated_env_dirs)} domain(s)")
        logger.info("=" * 80)
        
        # Step 1: Load all tools from generated environments
        logger.info("Step 1: Loading tools from generated environments...")
        tools_data = self._load_tools_from_environments(generated_env_dirs, domain_graph_dir)
        
        if not tools_data:
            logger.error("No valid tools found in the provided environments")
            return {}
        
        logger.info(f"Loaded {len(tools_data)} tools from {len(set(t['domain'] for t in tools_data))} domain(s)")
        
        # Step 2: Generate embeddings for all parameters
        logger.info("Step 2: Generating parameter embeddings...")
        param_embeddings = self._generate_parameter_embeddings(tools_data)
        
        # Step 3: Build dependency graph based on parameter similarity
        logger.info("Step 3: Building dependency graph...")
        graph = self._build_dependency_graph(tools_data, param_embeddings)
        
        # Step 4: Save outputs
        logger.info("Step 4: Saving outputs...")
        output_data = self._save_outputs(
            tools_data, graph, param_embeddings, output_dir, graph_name
        )
        
        logger.info("=" * 80)
        logger.info(f"Task graph building completed!")
        logger.info(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        logger.info("=" * 80)
        
        return output_data
    
    def _load_tools_from_environments(
        self,
        generated_env_dirs: List[str],
        domain_graph_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Load tools from generated environments and match with domain graph definitions.
        
        Args:
            generated_env_dirs: List of paths to generated environment directories
            domain_graph_dir: Path to domain graph directory
            
        Returns:
            List of tool data dictionaries
        """
        tools_data = []
        domain_graph_path = Path(domain_graph_dir)
        
        for env_dir_str in generated_env_dirs:
            env_dir = Path(env_dir_str)
            
            if not env_dir.exists():
                logger.warning(f"Environment directory does not exist: {env_dir}")
                continue
            
            # Find the domain name from environment_metadata.json
            metadata_file = env_dir / "environment_metadata.json"
            if not metadata_file.exists():
                logger.warning(f"No environment_metadata.json found in {env_dir}")
                continue
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            domain_name = metadata.get("domain")
            if not domain_name:
                logger.warning(f"No domain name found in {metadata_file}")
                continue
            
            # Load validation report to get successful tools
            validation_file = env_dir / "validation_report.json"
            if not validation_file.exists():
                logger.warning(f"No validation_report.json found in {env_dir}")
                continue
            
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_report = json.load(f)
            
            # Get list of successfully validated tools
            validation_results = validation_report.get("validation_results", {})
            successful_tools = [tool_name for tool_name, success in validation_results.items() if success]
            
            if not successful_tools:
                logger.warning(f"No successful tools found in {env_dir}")
                continue
            
            logger.info(f"Found {len(successful_tools)} successful tools in domain '{domain_name}'")
            
            # Load tool definitions from domain graph
            domain_file = domain_graph_path / "domains" / f"{domain_name}.json"
            if not domain_file.exists():
                logger.warning(f"Domain definition file not found: {domain_file}")
                continue
            
            with open(domain_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            # Get domain description
            domain_description = domain_data.get("description", "")
            
            # Extract tools that are in the successful list
            domain_tools = domain_data.get("tools", [])
            for tool in domain_tools:
                tool_name = tool.get("name")
                if tool_name in successful_tools:
                    tool_data = {
                        "name": tool_name,
                        "domain": domain_name,
                        "domain_description": domain_description,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "returns": tool.get("returns", {})
                    }
                    tools_data.append(tool_data)
                    logger.debug(f"Loaded tool: {tool_name} from domain {domain_name}")
        
        return tools_data
    
    def _generate_parameter_embeddings(
        self,
        tools_data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate embeddings for input and output parameters of all tools.
        
        Args:
            tools_data: List of tool data dictionaries
            
        Returns:
            Dictionary mapping tool names to their parameter embeddings
        """
        param_embeddings = {}
        
        for tool in tqdm(tools_data, desc="Generating parameter embeddings"):
            tool_name = tool["name"]
            
            try:
                # Generate embeddings for input parameters
                input_embeddings = {}
                parameters = tool.get("parameters", {})
                
                for param_name, param_info in parameters.items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    
                    # Create a rich description for embedding
                    full_desc = f"{param_type}: {param_desc}" if param_desc else param_type
                    
                    if full_desc:
                        embedding = self.embeddings.embed_text(full_desc)
                        input_embeddings[param_name] = {
                            "description": param_desc,
                            "type": param_type,
                            "embedding": embedding
                        }
                
                # Generate embeddings for output parameters
                output_embeddings = {}
                returns = tool.get("returns", {})
                
                # Handle different return formats
                if isinstance(returns, dict):
                    # Check if returns has 'properties' (structured return)
                    properties = returns.get("properties", {})
                    if properties:
                        for return_name, return_info in properties.items():
                            return_desc = return_info.get("description", "")
                            return_type = return_info.get("type", "")
                            
                            # Create a rich description for embedding
                            full_desc = f"{return_type}: {return_desc}" if return_desc else return_type
                            
                            if full_desc:
                                embedding = self.embeddings.embed_text(full_desc)
                                output_embeddings[return_name] = {
                                    "description": return_desc,
                                    "type": return_type,
                                    "embedding": embedding
                                }
                    else:
                        # Simple return with just type and description
                        return_desc = returns.get("description", "")
                        return_type = returns.get("type", "")
                        
                        full_desc = f"{return_type}: {return_desc}" if return_desc else return_type
                        
                        if full_desc:
                            embedding = self.embeddings.embed_text(full_desc)
                            output_embeddings["return_value"] = {
                                "description": return_desc,
                                "type": return_type,
                                "embedding": embedding
                            }
                
                param_embeddings[tool_name] = {
                    "inputs": input_embeddings,
                    "outputs": output_embeddings
                }
                
                logger.debug(f"Generated embeddings for {tool_name}: "
                           f"{len(input_embeddings)} inputs, {len(output_embeddings)} outputs")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {tool_name}: {e}")
                param_embeddings[tool_name] = {"inputs": {}, "outputs": {}}
        
        return param_embeddings
    
    def _build_dependency_graph(
        self,
        tools_data: List[Dict[str, Any]],
        param_embeddings: Dict[str, Dict[str, Any]]
    ) -> nx.DiGraph:
        """
        Build directed dependency graph based on parameter similarity.
        
        Edge direction: tool_with_output -> tool_with_input
        Edge weight: number of matching parameter pairs
        
        Args:
            tools_data: List of tool data dictionaries
            param_embeddings: Parameter embeddings for all tools
            
        Returns:
            Directed graph with tools as nodes
        """
        graph = nx.DiGraph()
        
        # Add all tools as nodes
        for tool in tools_data:
            graph.add_node(
                tool["name"],
                domain=tool["domain"],
                domain_description=tool.get("domain_description", ""),
                description=tool["description"],
                parameters=tool["parameters"],
                returns=tool["returns"]
            )
        
        # Calculate parameter similarities and add edges
        tool_names = [tool["name"] for tool in tools_data]
        edge_count = 0
        total_comparisons = len(tool_names) * (len(tool_names) - 1)
        
        logger.info(f"Comparing {total_comparisons} tool pairs for parameter dependencies...")
        
        with tqdm(total=total_comparisons, desc="Building edges") as pbar:
            for tool1_name in tool_names:
                for tool2_name in tool_names:
                    if tool1_name == tool2_name:
                        continue
                    
                    # Check if outputs of tool1 match inputs of tool2
                    matching_pairs = self._find_matching_parameters(
                        param_embeddings.get(tool1_name, {}).get("outputs", {}),
                        param_embeddings.get(tool2_name, {}).get("inputs", {})
                    )
                    
                    if matching_pairs:
                        # TODO: edge weight should be similarity sum
                        weight = len(matching_pairs)
                        graph.add_edge(
                            tool1_name,
                            tool2_name,
                            weight=weight,
                            matching_pairs=matching_pairs
                        )
                        edge_count += 1
                        
                        logger.debug(
                            f"Edge: {tool1_name} -> {tool2_name} "
                            f"(weight={weight}, pairs={matching_pairs})"
                        )
                    
                    pbar.update(1)
        
        logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {edge_count} edges")
        
        # Calculate and log graph statistics
        if edge_count > 0:
            avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            logger.info(f"Average degree: {avg_degree:.2f}")
            logger.info(f"Density: {nx.density(graph):.4f}")
        
        return graph
    
    def _find_matching_parameters(
        self,
        output_params: Dict[str, Dict[str, Any]],
        input_params: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        """
        Find matching parameter pairs between outputs and inputs.
        Each output parameter can only match to ONE input parameter (the one with highest similarity).
        
        Args:
            output_params: Output parameters with embeddings
            input_params: Input parameters with embeddings
            
        Returns:
            List of (output_param, input_param, similarity) tuples
        """
        matching_pairs = []
        
        for out_name, out_data in output_params.items():
            out_embedding = out_data.get("embedding")
            if out_embedding is None:
                continue
            
            # Find the best matching input parameter for this output parameter
            best_match = None
            best_similarity = -1.0
            
            for in_name, in_data in input_params.items():
                in_embedding = in_data.get("embedding")
                if in_embedding is None:
                    continue
                
                # Calculate cosine similarity
                similarity = self.embeddings.cosine_similarity(out_embedding, in_embedding)
                
                # Track the best match
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = in_name
            
            # Only add the match if it exceeds the threshold
            if best_match is not None and best_similarity >= self.similarity_threshold:
                matching_pairs.append((out_name, best_match, float(best_similarity)))
        
        return matching_pairs
    
    def _save_outputs(
        self,
        tools_data: List[Dict[str, Any]],
        graph: nx.DiGraph,
        param_embeddings: Dict[str, Dict[str, Any]],
        output_dir: str,
        graph_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save task graph and related data to files.
        
        Args:
            tools_data: List of tool data dictionaries
            graph: Dependency graph
            param_embeddings: Parameter embeddings
            output_dir: Output directory
            graph_name: Optional name for the graph
            
        Returns:
            Dictionary with output metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine graph name
        if graph_name is None:
            graph_name = "task_graph"
        
        # Save graph data
        graph_data = {
            "name": graph_name,
            "nodes": [
                {
                    "id": node,
                    **data
                }
                for node, data in graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1),
                    "matching_pairs": data.get("matching_pairs", [])
                }
                for u, v, data in graph.edges(data=True)
            ],
            "statistics": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "weakly_connected_components": nx.number_weakly_connected_components(graph),
                "strongly_connected_components": nx.number_strongly_connected_components(graph)
            },
            "metadata": {
                "similarity_threshold": self.similarity_threshold,
                "domains": list(set(tool["domain"] for tool in tools_data))
            }
        }
        
        with open(output_path / "task_graph.json", 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved task graph to {output_path / 'task_graph.json'}")
        
        # Save embeddings (optional, for analysis)
        embeddings_data = {
            "tool_embeddings": param_embeddings,
            "embedding_config": {
                "model": self.embeddings.model,
                "dimensions": self.embeddings.dimensions
            }
        }
        
        with open(output_path / "embeddings.json", 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved embeddings to {output_path / 'embeddings.json'}")
        
        # Generate visualization
        try:
            self._visualize_graph(graph, output_path / "task_graph_visualization.png")
            logger.info(f"Saved visualization to {output_path / 'task_graph_visualization.png'}")
        except Exception as e:
            logger.warning(f"Failed to generate graph visualization: {e}")
        
        return {
            "graph_name": graph_name,
            "output_directory": str(output_path),
            "statistics": graph_data["statistics"],
            "metadata": graph_data["metadata"]
        }
    
    def _visualize_graph(self, graph: nx.DiGraph, output_path: Path):
        """
        Generate directed graph visualization.
        
        Args:
            graph: Directed graph
            output_path: Path to save visualization
        """
        if graph.number_of_nodes() == 0:
            logger.warning("Cannot visualize empty graph")
            return
        
        plt.figure(figsize=(16, 12))
        
        # Use hierarchical layout for DAG if possible
        try:
            if nx.is_directed_acyclic_graph(graph):
                # Try to use graphviz_layout for better DAG visualization
                try:
                    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
                except:
                    # Fall back to spring layout
                    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
            else:
                pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
        # Color nodes by domain
        domains = list(set(nx.get_node_attributes(graph, 'domain').values()))
        domain_colors = plt.cm.Set3(np.linspace(0, 1, len(domains)))
        domain_color_map = {domain: domain_colors[i] for i, domain in enumerate(domains)}
        
        node_colors = [
            domain_color_map.get(graph.nodes[node].get('domain', ''), 'gray')
            for node in graph.nodes()
        ]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges with varying thickness based on weight
        edges = graph.edges()
        weights = [graph[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [1 + 3 * (w / max_weight) for w in weights]
        
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edges,
            width=edge_widths,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            arrowstyle='->'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title(
            f"Task Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n"
            f"DAG: {nx.is_directed_acyclic_graph(graph)}",
            fontsize=14,
            fontweight='bold'
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

