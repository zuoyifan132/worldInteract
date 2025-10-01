"""
Database schema generator that creates JSON-based database schemas from API collections.
"""
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from tenacity import RetryError, retry, stop_after_attempt, wait_fixed

from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.parser_utils import extract_json_from_text


logger = logging.getLogger(__name__)


class SchemaGenerator:
    """Generates database schemas from API collections using LLM."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize schema generator.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.model_config = self.config_manager.get_model_config("schema_generation")
    
    def generate_schema(self, api_collection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate database schema from API collection.
        
        Args:
            api_collection: API collection containing tools for a domain
            
        Returns:
            Generated database schema
        """
        domain = api_collection.get("domain", "unknown")
        tools = api_collection.get("tools", [])
        
        logger.info(f"Generating schema for domain: {domain}")
        
        # Get domain-specific configuration
        domain_config = self.config_manager.get_domain_config(domain)
        
        try:
            schema = self._generate_schema_with_llm(domain, tools, domain_config)
        except RetryError as e:
            logger.error(f"Failed to generate schema for domain: {domain}, error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate schema for domain: {domain}, error: {e}")
            raise
        
        logger.info(f"Successfully generated schema for domain: {domain}")
        return schema
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2)
    )
    def _generate_schema_with_llm(
        self, 
        domain: str, 
        tools: List[Dict[str, Any]], 
        domain_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate schema using LLM with retry mechanism.
        
        Args:
            domain: Domain name
            tools: List of tools in the domain
            domain_config: Domain-specific configuration
            
        Returns:
            Generated schema
        """
        system_prompt = self._create_schema_system_prompt()
        user_prompt = self._create_schema_user_prompt(domain, tools, domain_config)
        
        try:
            thinking_content, answer_text, function_calls = generate(
                model_key=self.model_config["model"],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=self.model_config.get("max_tokens", 4000)
            )
            
            # Extract JSON response
            json_schema = extract_json_from_text(answer_text)
            schema = json.loads(json_schema)
            
            # Validate the schema structure
            self._validate_schema(schema)

            logger.info(f"Schema generated: {json.dumps(schema, indent=2)}")
            
            return schema
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse schema JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate schema: {e}")
            raise
    
    def _create_schema_system_prompt(self) -> str:
        """Create system prompt for schema generation."""
        return """You are an expert database designer specializing in creating JSON-based database schemas for function-calling environments.

Your task is to design a lightweight, in-memory database schema that supports the given API collection. The schema should follow these principles:

1. **Lightweight Design**: Use JSON objects as tables/collections, following τ-bench principles
2. **In-Memory Operations**: Designed for fast in-memory read/write operations
3. **Relationship Support**: Include proper relationships between entities
4. **Tool Compatibility**: Schema must support all provided tools' operations

Schema Structure Requirements:
- Each "table" is a JSON object with records keyed by unique IDs
- Include metadata fields (created_at, updated_at, etc.) where appropriate
- Support both read and write operations for the tools
- Maintain referential integrity through ID relationships
- Keep the schema simple but comprehensive

Output Format:
Return ONLY a valid JSON object representing the schema. Each top-level key represents a "table" with the following structure:
```json
{
  "table_name": {
    "description": "Table description",
    "primary_key": "field_name",
    "fields": {
      "field_name": {
        "type": "string|integer|float|boolean|array|object",
        "description": "Field description",
        "required": true|false,
        "default": "default_value (optional)"
      }
    },
    "relationships": {
      "field_name": {
        "type": "one_to_one|one_to_many|many_to_many",
        "table": "related_table_name",
        "field": "related_field_name"
      }
    }
  }
}
```"""
    
    def _create_schema_user_prompt(
        self, 
        domain: str, 
        tools: List[Dict[str, Any]], 
        domain_config: Dict[str, Any]
    ) -> str:
        """Create user prompt for schema generation."""
        tools_info = []
        for tool in tools:
            tool_info = f"- **{tool['name']}**: {tool['description']}\n"
            if 'parameters' in tool:
                tool_info += f"  Parameters: {list(tool['parameters'].keys())}\n"
            if 'returns' in tool:
                tool_info += f"  Returns: {tool['returns']}\n"
            tools_info.append(tool_info)
        
        domain_tables = domain_config.get("schema_tables", [])
        domain_hint = f"\nSuggested tables for this domain: {domain_tables}" if domain_tables else ""
        
        return f"""Design a database schema for the **{domain}** domain with the following tools:

{chr(10).join(tools_info)}

Domain: {domain}
{domain_hint}

Requirements:
1. Analyze the tools to understand what data entities are needed
2. Design tables/collections that support all tool operations
3. Include proper relationships between entities
4. Ensure the schema supports both read and write operations
5. Follow the JSON-based lightweight design principles

Generate a comprehensive schema that enables these tools to operate on realistic data."""
    
    def _validate_schema(self, schema: Dict[str, Any]) -> None:
        """
        Validate the generated schema structure.
        
        Args:
            schema: Generated schema to validate
            
        Raises:
            ValueError: If schema structure is invalid
        """
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
        
        for table_name, table_def in schema.items():
            if not isinstance(table_def, dict):
                raise ValueError(f"Table definition for '{table_name}' must be a dictionary")
            
            required_keys = ["description", "fields"]
            for key in required_keys:
                if key not in table_def:
                    raise ValueError(f"Table '{table_name}' missing required key: {key}")
            
            # Validate fields structure
            fields = table_def.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(f"Fields in table '{table_name}' must be a dictionary")
            
            for field_name, field_def in fields.items():
                if not isinstance(field_def, dict):
                    raise ValueError(f"Field definition for '{field_name}' must be a dictionary")
                
                if "type" not in field_def:
                    raise ValueError(f"Field '{field_name}' missing required 'type' key")
        
        logger.info("Schema validation passed")
    
    def save_schema(self, schema: Dict[str, Any], domain: str, output_dir: Optional[str] = None) -> str:
        """
        Save generated schema to file.
        
        Args:
            schema: Generated schema
            domain: Domain name
            output_dir: Output directory (defaults to data/generated/domains/)
            
        Returns:
            Path to saved schema file
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            output_dir = project_root / "data" / "generated" / "domains" / domain
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        schema_file = output_path / "schema.json"
        
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Schema saved to: {schema_file}")
        return str(schema_file)
    
    def load_schema(self, domain: str, schema_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load existing schema from file.
        
        Args:
            domain: Domain name
            schema_dir: Directory containing schema files
            
        Returns:
            Loaded schema
        """
        if schema_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            schema_dir = project_root / "data" / "generated" / "domains" / domain
        
        schema_file = Path(schema_dir) / "schema.json"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        logger.info(f"Schema loaded from: {schema_file}")
        return schema

