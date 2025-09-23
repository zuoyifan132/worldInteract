"""
Environment manager that orchestrates the entire environment construction pipeline.
"""

import json
import logging
import random
from typing import Dict, Any, Optional
from pathlib import Path

from worldInteract.core.schema_generator import SchemaGenerator
from worldInteract.core.tool_generator import ToolGenerator
from worldInteract.core.validator import ToolValidator
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.model_manager import generate
from worldInteract.utils.parser_utils import extract_json_from_text


logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Orchestrates the complete environment construction pipeline."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize environment manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.schema_generator = SchemaGenerator(config_dir)
        self.tool_generator = ToolGenerator(config_dir)
        self.validator = ToolValidator(config_dir)
        
        # Get state generation config
        self.state_config = self.config_manager.get_model_config("state_generation")
    
    def create_environment(
        self,
        api_collection_path: str,
        output_dir: Optional[str] = None,
        validate_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Create a complete environment from API collection.
        
        Args:
            api_collection_path: Path to API collection JSON file
            output_dir: Output directory for generated files. If None, automatically 
                       determined from domain field: data/generated/domains/{domain}/
            validate_tools: Whether to validate generated tools
            
        Returns:
            Environment information dictionary
        """
        # Load API collection
        api_collection = self.load_api_collection(api_collection_path)
        domain = api_collection.get("domain", "unknown")
        
        logger.info(f"Creating environment for domain: {domain}")
        
        try:
            # Step 1: Generate database schema
            logger.info("Step 1: Generating database schema...")
            schema = self.schema_generator.generate_schema(api_collection)
            
            # Step 2: Generate initial state
            logger.info("Step 2: Generating initial database state...")
            initial_state = self.generate_initial_state(schema, api_collection)
            
            # Step 3: Generate tool implementations
            # skip 
            logger.info("Step 3: Generating tool implementations...")
            tools = self.tool_generator.generate_tools(api_collection, schema, initial_state)
            
            # Step 4: Validate tools (if requested)
            # TODO: using ReAct agent mode to validate and regenerate tools if needed, set maximum ReAct rouds
            # if exceeded the maximum ReAct rounds, skip this domain
            validation_results = {}
            if validate_tools:
                logger.info("Step 4: Validating generated tools...")
                validation_results = self.validator.validate_tools(
                    tools, schema, initial_state, api_collection
                )
                
                # Regenerate failed tools (up to 3 attempts)
                # failed_tools = [name for name, passed in validation_results.items() if not passed]
                # if failed_tools:
                #     logger.warning(f"Regenerating {len(failed_tools)} failed tools...")
                #     tools, validation_results = self._regenerate_failed_tools(
                #         failed_tools, api_collection, schema, initial_state, tools, validation_results
                #     )
            
            # Step 5: Save all generated components
            if output_dir is None:
                project_root = Path(__file__).parent.parent.parent.parent
                output_dir = project_root / "data" / "generated" / "domains" / domain
            
            self._save_environment(
                domain, schema, initial_state, tools, validation_results, output_dir
            )
            
            environment_info = {
                "domain": domain,
                "schema": schema,
                "initial_state": initial_state,
                "tools": tools,
                "validation_results": validation_results,
                "output_dir": str(output_dir)
            }
            
            logger.info(f"Environment creation completed for domain: {domain}")
            return environment_info
            
        except Exception as e:
            logger.error(f"Failed to create environment for domain {domain}: {e}")
            raise
    
    def load_api_collection(self, api_collection_path: str) -> Dict[str, Any]:
        """
        Load API collection from file.
        
        Args:
            api_collection_path: Path to API collection JSON file
            
        Returns:
            Loaded API collection
        """
        collection_path = Path(api_collection_path)
        
        if not collection_path.exists():
            raise FileNotFoundError(f"API collection file not found: {api_collection_path}")
        
        with open(collection_path, 'r', encoding='utf-8') as f:
            api_collection = json.load(f)
        
        logger.info(f"Loaded API collection: {api_collection.get('domain', 'unknown')}")
        return api_collection
    
    def generate_initial_state(
        self,
        schema: Dict[str, Any],
        api_collection: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate initial database state based on schema.
        
        Args:
            schema: Database schema
            api_collection: API collection for context
            
        Returns:
            Generated initial state
        """
        domain = api_collection.get("domain", "unknown")
        logger.info(f"Generating initial state for domain: {domain}")
        
        system_prompt = self._create_state_generation_system_prompt()
        user_prompt = self._create_state_generation_user_prompt(schema, api_collection)
        
        try:
            thinking_content, answer_text, function_calls = generate(
                model_key=self.state_config["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.state_config.get("temperature", 0.4),
                max_tokens=self.state_config.get("max_tokens", 3000)
            )

            # Extract JSON response
            json_state = extract_json_from_text(answer_text)
            
            # Parse the JSON response
            initial_state = json.loads(json_state)
            
            # Validate the state structure
            self._validate_initial_state(initial_state, schema)

            logger.info(f"Initial state generated: {json.dumps(initial_state, indent=2)}")
            
            logger.info("Successfully generated initial state")
            return initial_state
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse initial state JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate initial state: {e}")
            raise
    
    def _create_state_generation_system_prompt(self) -> str:
        """Create system prompt for initial state generation."""
        return """You are a database state generator that creates realistic initial data for testing environments.

Your task is to generate diverse, realistic initial data that follows the provided schema and supports comprehensive testing of the tools.

Requirements:
1. **Schema Compliance**: Follow the exact schema structure and field types
2. **Realistic Data**: Generate data that resembles real-world usage
3. **Diversity**: Include varied data to test different scenarios
4. **Relationships**: Maintain proper relationships between entities
5. **Completeness**: Provide enough data for meaningful tool testing

Data Generation Guidelines:
- Generate desired number of records per table (depending on table importance)
- Use realistic names, dates, IDs, and values
- Include both typical and edge cases in the data
- Maintain referential integrity across related tables
- Include some records with different states/statuses

Output Format:
Return ONLY a valid JSON object where each top-level key matches a table name from the schema:
```json
{
  "table_name": {
    "record_id_1": {
      "field1": "value1",
      "field2": "value2",
      ...
    },
    "record_id_2": { ... }
  },
  "another_table": {
    ...
  }
}
```"""
    
    def _create_state_generation_user_prompt(
        self,
        schema: Dict[str, Any],
        api_collection: Dict[str, Any]
    ) -> str:
        """Create user prompt for initial state generation."""
        domain = api_collection.get("domain", "unknown")
        
        # Get domain configuration for guidance
        domain_config = self.config_manager.get_domain_config(domain)
        min_records = domain_config.get("min_records_per_table", 5)
        max_records = domain_config.get("max_records_per_table", 20)

        # generate random number of records per table
        records_per_table = random.randint(min_records, max_records)
        
        # Format schema information
        schema_info = []
        for table_name, table_def in schema.items():
            fields = table_def.get("fields", {})
            field_names = list(fields.keys())
            schema_info.append(f"- **{table_name}**: {table_def.get('description', '')}")
            schema_info.append(f"  Fields: {field_names}")
            
            # Include relationship information
            relationships = table_def.get("relationships", {})
            if relationships:
                rel_info = []
                for field, rel in relationships.items():
                    rel_info.append(f"{field} -> {rel.get('table', '')}")
                schema_info.append(f"  Relationships: {rel_info}")
        
        return f"""Generate realistic initial data for the **{domain}** domain.

**Database Schema**:
{chr(10).join(schema_info)}

**Domain**: {domain}
**Records per table**: {records_per_table}

**Requirements**:
1. Generate diverse, realistic data appropriate for the {domain} domain
2. Follow the exact schema structure and data types
3. Maintain relationships between tables where specified
4. Include realistic IDs, names, dates, and status values
5. Ensure data supports comprehensive testing of domain tools
6. Include some edge cases and varied scenarios

Generate a complete initial database state that provides a solid foundation for tool testing in this domain."""
    
    def _validate_initial_state(self, initial_state: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """
        Validate the generated initial state against the schema.
        
        Args:
            initial_state: Generated initial state
            schema: Database schema
            
        Raises:
            ValueError: If state doesn't match schema
        """
        # Check that all schema tables are present
        for table_name in schema.keys():
            if table_name not in initial_state:
                raise ValueError(f"Table '{table_name}' missing from initial state")
            
            if not isinstance(initial_state[table_name], dict):
                raise ValueError(f"Table '{table_name}' must be a dictionary")
        
        # Check that no extra tables are present
        for table_name in initial_state.keys():
            if table_name not in schema:
                logger.warning(f"Extra table '{table_name}' in initial state (not in schema)")
        
        logger.info("Initial state validation passed")
    
    def _regenerate_failed_tools(
        self,
        failed_tools: list,
        api_collection: Dict[str, Any],
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        tools: Dict[str, str],
        validation_results: Dict[str, bool]
    ) -> tuple:
        """
        Regenerate failed tools with up to 3 attempts each.
        
        Args:
            failed_tools: List of tool names that failed validation
            api_collection: Original API collection
            schema: Database schema
            initial_state: Initial database state
            tools: Current tool implementations
            validation_results: Current validation results
            
        Returns:
            Tuple of (updated_tools, updated_validation_results)
        """
        max_attempts = 3
        
        for tool_name in failed_tools:
            logger.info(f"Attempting to regenerate tool: {tool_name}")
            
            # Find the tool description
            tool_desc = None
            for tool in api_collection.get("tools", []):
                if tool.get("name") == tool_name:
                    tool_desc = tool
                    break
            
            if not tool_desc:
                logger.error(f"Tool description not found for: {tool_name}")
                continue
            
            # Try regenerating up to max_attempts times
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Regeneration attempt {attempt + 1}/{max_attempts} for {tool_name}")
                    
                    # Generate new tool implementation
                    new_tool_code = self.tool_generator._generate_tool_with_llm(
                        tool_desc, schema, api_collection.get("domain", "unknown"), initial_state
                    )
                    
                    # Test the new implementation
                    is_valid = self.validator._validate_single_tool(
                        tool_name, new_tool_code, tool_desc, schema, initial_state
                    )
                    
                    if is_valid:
                        tools[tool_name] = new_tool_code
                        validation_results[tool_name] = True
                        logger.info(f"Successfully regenerated tool: {tool_name}")
                        break
                    else:
                        logger.warning(f"Regenerated tool {tool_name} still failed validation (attempt {attempt + 1})")
                        
                except Exception as e:
                    logger.error(f"Error regenerating tool {tool_name} (attempt {attempt + 1}): {e}")
            
            if not validation_results.get(tool_name, False):
                logger.error(f"Failed to regenerate tool {tool_name} after {max_attempts} attempts")
        
        return tools, validation_results
    
    def _save_environment(
        self,
        domain: str,
        schema: Dict[str, Any],
        initial_state: Dict[str, Any],
        tools: Dict[str, str],
        validation_results: Dict[str, bool],
        output_dir: Path
    ) -> None:
        """Save all environment components to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save schema
        self.schema_generator.save_schema(schema, domain, output_path)
        
        # Save initial state
        state_file = output_path / "initial_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(initial_state, f, indent=2, ensure_ascii=False)
        
        # Save tools
        self.tool_generator.save_tools(tools, domain, output_path)
        
        # Save validation report
        if validation_results:
            self.validator.save_validation_report(validation_results, domain, output_path)
        
        # Save environment metadata
        metadata = {
            "domain": domain,
            "schema_file": "schema.json",
            "initial_state_file": "initial_state.json",
            "tools_dir": "tools/",
            "tools_file": "tools.py",
            "validation_report": "validation_report.json",
            "created_at": str(__import__('datetime').datetime.now()),
            "tool_count": len(tools),
            "validation_passed": sum(1 for result in validation_results.values() if result) if validation_results else 0
        }
        
        metadata_file = output_path / "environment_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Environment saved to: {output_path}")
    
    def load_environment(self, domain: str, environment_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load an existing environment.
        
        Args:
            domain: Domain name
            environment_dir: Directory containing environment files
            
        Returns:
            Environment information dictionary
        """
        if environment_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            environment_dir = project_root / "data" / "generated" / "domains" / domain
        
        env_path = Path(environment_dir)
        
        if not env_path.exists():
            raise FileNotFoundError(f"Environment directory not found: {env_path}")
        
        # Load schema
        schema = self.schema_generator.load_schema(domain, env_path)
        
        # Load initial state
        state_file = env_path / "initial_state.json"
        with open(state_file, 'r', encoding='utf-8') as f:
            initial_state = json.load(f)
        
        # Load tools
        tools = self.tool_generator.load_tools(domain, env_path / "tools")
        
        # Load validation results (if available)
        validation_results = {}
        validation_file = env_path / "validation_report.json"
        if validation_file.exists():
            with open(validation_file, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
                validation_results = validation_data.get("validation_results", {})
        
        environment_info = {
            "domain": domain,
            "schema": schema,
            "initial_state": initial_state,
            "tools": tools,
            "validation_results": validation_results,
            "environment_dir": str(env_path)
        }
        
        logger.info(f"Environment loaded for domain: {domain}")
        return environment_info

