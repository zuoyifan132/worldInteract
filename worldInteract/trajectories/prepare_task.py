"""
Task preparation module for generating user queries from random walks.
Uses LLM to generate coherent user tasks based on DAG structure and edges.
"""

import json
import logging
from multiprocessing import Value
from typing import Dict, Any, List, Tuple
from textwrap import dedent

from worldInteract.utils.camel_generator import generate
from worldInteract.utils.config_manager import config_manager


logger = logging.getLogger(__name__)


class TaskPreparer:
    """Prepares user tasks/queries for each node in a random walk."""
    
    def __init__(self):
        """Initialize TaskPreparer."""
        self.config_key = "trajectory_generation"
        self.config = config_manager.get_environment_config(self.config_key)
        self.model_config = config_manager.get_model_config(self.config_key)
        
    def generate_user_queries(
        self,
        random_walk: Dict[str, Any],
        domain_info: Dict[str, Any],
        initial_state: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate user queries for each node in the random walk sequence.
        
        Args:
            random_walk: Random walk data with sequence, nodes, edges, dag_structure
            domain_info: Domain information with all tool descriptions
            initial_state: Environment initial state (database, files, etc.)
            
        Returns:
            List of user queries with node information:
            [
                {
                    "node_id": "ls",
                    "user_query": "Can you show me what files are in my current directory?",
                    "node_description": "...",
                    ...
                },
                ...
            ]
        """
        logger.info(f"Generating user queries for random walk {random_walk['id']}")
        
        # Create prompt for LLM
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(random_walk, domain_info, initial_state)
        
        # Generate queries using LLM
        try:
            thinking_content, answer_text, function_calls = generate(
                config_key=self.config_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 8192)
            )
            
            # Extract queries from response
            queries = self._extract_queries(answer_text, random_walk)
            
            if len(queries) != len(random_walk["sequence"]):
                logger.warning(
                    f"Generated {len(queries)} queries but expected {len(random_walk['sequence'])}"
                )
            
            logger.info(f"Successfully generated {len(queries)} user queries")
            return queries
            
        except Exception as e:
            logger.error(f"Failed to generate user queries: {e}")
            raise
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for query generation."""
        return dedent(
            """You are a task scenario generation expert. Your task is to generate coherent user queries or tasks based on tool invocation workflows (DAG structure).

            ## Your Goals:
            - Generate natural and reasonable user queries for each node (tool)
            - Ensure the entire workflow is coherent and aligned with real-world usage scenarios
            - Queries should guide the agent to invoke the corresponding tools
            - Consider dependencies and data flow between tools
            - Use environment initial state to generate SPECIFIC and CONCRETE queries with real data

            ## Output Format:
            Please generate a user query for each tool using the following JSON format:

            ```json
            [
                {
                    "node_id": "tool_name_1",
                    "user_query": "Specific user query or task description"
                },
                {
                    "node_id": "tool_name_2", 
                    "user_query": "Next query based on previous results"
                }
            ]
            ```

            ## Requirements:
            1. **Start node queries** should be independent initial tasks
            2. **Subsequent node queries** should naturally depend on or continue from previous operations
            3. **Parallel nodes** can be relatively independent but related tasks
            4. Each query should be clear, specific, and aligned with real-world scenarios
            5. The entire workflow should tell a complete story or task sequence
            6. Queries can be simple (single-step operations) or complex (requiring multi-step ReAct)
            7. **If initial state is provided**, reference specific entities (files, users, records, etc.) from it
            
            ## Notes:
            - Do not directly specify tool names in queries
            - Describe user intent in natural language
            - Ensure query order matches the sequence
            - Make queries concrete by using actual data from the environment state when available"""
        )
    
    def _create_user_prompt(
        self,
        random_walk: Dict[str, Any],
        domain_info: Dict[str, Any],
        initial_state: Dict[str, Any] = None
    ) -> str:
        """Create user prompt with random walk, domain information, and initial state."""
        walk_id = random_walk["id"]
        sequence = random_walk["sequence"]
        nodes = random_walk["nodes"]
        edges = random_walk["edges"]
        dag_structure = random_walk["dag_structure"]
        
        # Format node information
        nodes_info = []
        for node in nodes:
            node_info = {
                "id": node["id"],
                "description": node["description"],
                "parameters": node["parameters"],
                "returns": node["returns"]
            }
            nodes_info.append(node_info)
        
        # Format edge information with examples
        edges_info = []
        for edge in edges:
            edge_info = {
                "source": edge["source"],
                "target": edge["target"],
                "matching_pairs": edge.get("matching_pairs", []),
                "example_usage": edge.get("example_usage", [])
            }
            edges_info.append(edge_info)
        
        # Format initial state information if provided
        initial_state_section = ""
        if initial_state:
            initial_state_section = f"""
            **Environment Initial State** (current state of the environment before execution):
            ```json
            {json.dumps(initial_state, indent=2, ensure_ascii=False)}
            ```
            
            **Important**: Use the initial state information to generate SPECIFIC and CONCRETE user queries.
            For example, if there are specific files, users, or records in the initial state, reference them in your queries.
            """
        
        return dedent(
            f"""Please generate user queries for the following random walk. This is a {random_walk['walk_type']} structure tool invocation workflow.

            **Random Walk ID**: {walk_id}
            **Domain**: {domain_info['domain']}
            **Domain Description**: {domain_info['description']}
            {initial_state_section}
            **Execution Sequence (in this order)**:
            {json.dumps(sequence, indent=2, ensure_ascii=False)}
            
            **DAG Structure** (for understanding dependencies between tools):
            ```json
            {json.dumps(dag_structure, indent=2, ensure_ascii=False)}
            ```
            
            **Node Information** (detailed description of each tool):
            ```json
            {json.dumps(nodes_info, indent=2, ensure_ascii=False)}
            ```
            
            **Edge Information** (data flow and usage examples between tools):
            ```json
            {json.dumps(edges_info, indent=2, ensure_ascii=False)}
            ```
            
            **Task**:
            Please generate a reasonable user query for each tool in the sequence (in order).
            - The first tool "{sequence[0]}" is the start node, generate an independent initial query
            - Subsequent tool queries should be based on previous operation results, forming a coherent task flow
            - Refer to example_usage in edges to understand relationships between tools
            - Ensure the entire workflow tells a complete and reasonable usage scenario
            {f"- Use concrete data from the initial state to make queries specific and realistic" if initial_state else ""}
            
            Please output the user query list in JSON format."""
        )
    
    def _extract_queries(
        self,
        response: str,
        random_walk: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract user queries from LLM response."""
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                queries_list = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}, trying alternative extraction")
                queries_list = self._fallback_extraction(response)
        else:
            # Fallback: try to parse the entire response as JSON
            try:
                queries_list = json.loads(response)
            except json.JSONDecodeError:
                logger.warning("No JSON found in response, trying alternative extraction")
                queries_list = self._fallback_extraction(response)
        
        # Enrich queries with node information
        enriched_queries = []
        sequence = random_walk["sequence"]
        nodes_map = {node["id"]: node for node in random_walk["nodes"]}
        
        for i, query_item in enumerate(queries_list):
            if i >= len(sequence):
                logger.warning(f"More queries than nodes in sequence, skipping extra query")
                break
                
            node_id = sequence[i]
            node_info = nodes_map.get(node_id, {})
            
            enriched_query = {
                "node_id": node_id,
                "user_query": query_item.get("user_query", ""),
                "domain": node_info.get("domain", ""),
                "domain_description": node_info.get("domain_description", ""),
                # "tool_description": node_info.get("description", ""),
                # "parameters": node_info.get("parameters", {}),
                # "returns": node_info.get("returns", {})
            }
            enriched_queries.append(enriched_query)
        
        # Missing queries error
        if len(enriched_queries) < len(sequence):
            logger.warning(f"Missing queries error: Generated fewer queries than expected")
            raise ValueError(f"Missing queries error: Generated fewer queries than expected")
        
        return enriched_queries
    
    def _fallback_extraction(self, response: str) -> List[Dict[str, Any]]:
        """Fallback method to extract queries from unstructured response."""
        # Simple pattern matching for node_id and query
        import re
        
        queries = []
        # Try to find patterns like "node_id": "xxx", "user_query": "yyy"
        pattern = r'"node_id":\s*"([^"]+)"[^}]*"user_query":\s*"([^"]+)"'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            queries.append({
                "node_id": match[0],
                "user_query": match[1]
            })
        
        return queries if queries else []

