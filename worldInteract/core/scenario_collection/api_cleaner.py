"""
API Cleaner for standardizing and cleaning raw API descriptions.
"""

import json
import re
import textwrap
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from loguru import logger
from difflib import SequenceMatcher
from tqdm import tqdm
from tenacity import RetryError

from worldInteract.core.scenario_collection import similarity_method
from worldInteract.utils.model_manager import generate
from worldInteract.utils.config_manager import config_manager
from worldInteract.utils.embedding import OpenAIEmbeddings
from worldInteract.utils.parser_utils import extract_json_from_text
from worldInteract.core.scenario_collection.similarity_method import SimilarityMethod


# Standardize type names mapping
TYPE_MAPPING = {
    "str": "string",
    "int": "integer",
    "bool": "boolean",
    "obj": "object",
    "arr": "array"
}
# Standardize description fields
DESC_FIELDS = ["description", "desc", "info"]
# Standardize type fields
TYPE_FIELDS = ["type", "datatype", "dtype"]


class APICleaner:
    """Cleans and standardizes raw API descriptions into a uniform format."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize API cleaner.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_manager = config_manager
        self.model_config = self.config_manager.get_model_config("scenario_collection")
        self.env_config = self.config_manager.get_environment_config("scenario_collection")
        
        # Initialize embeddings for duplicate detection
        try:
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}. Duplicate detection will use text similarity.")
            self.embeddings = None
        
        # Configuration
        self.required_fields = self.env_config.get("required_fields", ["name", "description", "parameters"])
        self.min_name_length = self.env_config.get("min_tool_name_length", 3)
        self.max_name_length = self.env_config.get("max_tool_name_length", 50)
        self.duplicate_threshold = self.env_config.get("duplicate_threshold", 0.8)
        self.min_description_length = self.env_config.get("description_min_length", 10)
        
        # Parse similarity method from config
        similarity_method_str = self.env_config.get("similarity_method", SimilarityMethod.get_default().value)
        try:
            # Handle None as a special case
            if similarity_method_str is None or str(similarity_method_str).lower() == "none":
                self.similarity_method = SimilarityMethod.NONE
            else:
                self.similarity_method = SimilarityMethod.from_string(similarity_method_str)
        except ValueError as e:
            logger.warning(f"{e}. Using default method.")
            self.similarity_method = SimilarityMethod.get_default()
        
        # If embedding_model is configured but embeddings are not available, fall back to sequence_matcher
        if self.similarity_method.is_embedding_based() and self.embeddings is None:
            logger.warning(f"similarity_method configured as '{self.similarity_method}' but embeddings unavailable. Falling back to '{SimilarityMethod.SEQUENCE_MATCHER}'")
            self.similarity_method = SimilarityMethod.SEQUENCE_MATCHER
        
        logger.info(f"Initialized API Cleaner with similarity method: {self.similarity_method}")
    
    def _load_raw_apis(self, raw_apis_path: str) -> List[Dict[str, Any]]:
        """
        Load raw APIs from a single file or directory of JSON files.
        
        Args:
            raw_apis_path: Path to JSON file or directory containing JSON files
            
        Returns:
            List of raw API dictionaries
        """
        apis_path = Path(raw_apis_path)
        all_apis = []
        
        if apis_path.is_file():
            # Single file
            logger.info(f"Loading APIs from single file: {apis_path}")
            with open(apis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            apis = data.get("apis", [])
            all_apis.extend(apis)
            logger.info(f"Loaded {len(apis)} APIs from {apis_path.name}")
            
        elif apis_path.is_dir():
            # Directory of JSON files
            logger.info(f"Loading APIs from directory: {apis_path}")
            json_files = list(apis_path.glob("*.json"))
            
            if not json_files:
                logger.warning(f"No JSON files found in directory: {apis_path}")
                return []
            
            for json_file in json_files:
                try:
                    logger.info(f"Loading APIs from: {json_file.name}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        apis = data.get("apis", [])
                    elif isinstance(data, list):
                        apis = data
                    all_apis.extend(apis)
                    logger.info(f"Loaded {len(apis)} APIs from {json_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load APIs from {json_file}: {e}")
                    continue
        else:
            raise FileNotFoundError(f"Path not found: {raw_apis_path}")
        
        logger.info(f"Total APIs loaded: {len(all_apis)}")
        return all_apis
    
    def clean_apis(self, raw_apis_path: str, output_path: str) -> Dict[str, Any]:
        """
        Clean and standardize raw APIs.
        
        Args:
            raw_apis_path: Path to raw APIs JSON file or directory containing JSON files
            output_path: Path to save cleaned APIs
            
        Returns:
            Dictionary with cleaned APIs and processing statistics
        """
        logger.info(f"Starting API cleaning from {raw_apis_path}")
        
        # Load raw APIs - support both single file and directory
        raw_apis = self._load_raw_apis(raw_apis_path)
        logger.info(f"Loaded {len(raw_apis)} raw APIs")
        
        # Processing statistics
        stats = {
            "total_input": len(raw_apis),
            "fixed": 0,
            "removed_duplicates": 0,
            "removed_invalid": 0,
            "final_count": 0
        }
        
        # Step 1: Fix and standardize APIs
        fixed_apis = []
        for api in tqdm(raw_apis, desc="Fixing APIs"):
            try:
                fixed_api = self.standardize_and_fix_api_format(api)
                if fixed_api:
                    fixed_apis.append(fixed_api)
                    stats["fixed"] += 1
                else:
                    stats["removed_invalid"] += 1
            except Exception as e:
                logger.error(f"Error fixing API {api.get('name', 'unknown')}: {e}")
                stats["removed_invalid"] += 1
        
        logger.info(f"Fixed {len(fixed_apis)} APIs")
        
        # Step 2: Remove duplicates
        unique_apis = self._remove_duplicates(fixed_apis)
        stats["removed_duplicates"] = len(fixed_apis) - len(unique_apis)
        stats["final_count"] = len(unique_apis)
        
        logger.info(f"Removed {stats['removed_duplicates']} duplicates, {len(unique_apis)} unique APIs remain")
        
        # Step 3: Enhance descriptions using LLM
        enhanced_apis = self._enhance_descriptions(unique_apis)
        
        # Prepare output
        output_data = {
            "metadata": {
                "source": raw_apis_path,
                "processing_stats": stats,
                "total_apis": len(enhanced_apis)
            },
            "apis": enhanced_apis
        }
        
        # Save output
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(enhanced_apis)} cleaned APIs to {output_path}")
        return output_data
    
    def standardize_and_fix_api_format(self, api: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fix and standardize a single API format.
        
        Args:
            api: Raw API dictionary
            
        Returns:
            Standardized API dictionary or None if invalid
        """
        # Extract name from various possible fields
        name = self._extract_name(api)
        if not name or len(name) < self.min_name_length or len(name) > self.max_name_length:
            return None
        
        # Extract description
        description = self._extract_description(api)
        if not description or len(description) < self.min_description_length:
            # Try to generate description using LLM
            # TODO: Parallel execution
            description = self._generate_description(name, api)
            logger.info(f"Generated new description for {name}: {description}")
            if not description:
                return None
        
        # Extract parameters
        parameters = self._extract_parameters(api)
        
        # Extract returns
        returns = self._extract_returns(api)
        
        # Create standardized format
        standardized = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        
        if returns:
            standardized["returns"] = returns
        
        return standardized
    
    def _extract_name(self, api: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from various possible fields."""
        name_fields = ["name", "function_name", "tool_name", "api_name", "function", "tool", "api"]
        
        for field in name_fields:
            if field in api and api[field]:
                name = str(api[field]).strip()
                if name:
                    # Clean name: remove special characters, convert to snake_case
                    name = re.sub(r'[^\w\s]', '', name)
                    name = re.sub(r'\s+', '_', name).lower()
                    return name
        
        return None
    
    def _extract_description(self, api: Dict[str, Any]) -> Optional[str]:
        """Extract description from various possible fields."""
        desc_fields = ["description", "desc", "summary", "info"]
        
        for field in desc_fields:
            if field in api and api[field]:
                desc = str(api[field]).strip()
                if len(desc) >= self.min_description_length:
                    return desc
        
        return None
    
    def _extract_parameters(self, api: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize parameters."""
        param_fields = ["parameters", "params", "arguments", "args", "inputs"]
        
        for field in param_fields:
            if field in api and isinstance(api[field], dict):
                params = api[field]

                if "properties" in params:
                    params = params["properties"]

                standardized_params = {}
                
                for param_name, param_info in params.items():
                    if isinstance(param_info, dict):
                        # Standardize parameter info
                        std_param = {}
                        
                        # Type
                        for type_field in TYPE_FIELDS:
                            if type_field in param_info:
                                param_type = str(param_info[type_field]).strip()
                                # Standardize type names
                                std_param["type"] = TYPE_MAPPING.get(param_type, param_type)
                                break
                        
                        # Description
                        for desc_field in DESC_FIELDS:
                            if desc_field in param_info:
                                std_param["description"] = str(param_info[desc_field]).strip()
                                break
                        
                        # Default value
                        if "default" in param_info:
                            std_param["default"] = param_info["default"]
                        
                        standardized_params[param_name] = std_param
                    else:
                        # Simple type specification
                        standardized_params[param_name] = {
                            "type": str(param_info),
                            "description": f"Parameter {param_name}"
                        }
                
                return standardized_params
        
        return {}
    
    def _extract_returns(self, api: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract return type specification."""
        return_fields = ["returns", "return", "output", "response"]
        
        for field in return_fields:
            if field in api and api[field]:
                return_info = api[field]
                if isinstance(return_info, dict):
                    return return_info
                else:
                    # Simple return type
                    return {
                        "type": str(return_info),
                        "description": "Function return value"
                    }
        
        return None
    
    def _generate_description(self, name: str, api: Dict[str, Any]) -> Optional[str]:
        """Generate description using LLM when missing."""
        try:
            system_prompt = textwrap.dedent(
                f"""You are an API documentation expert. Generate a clear and concise tool description based on the tool name, parameter names, and parameter descriptions.

                Requirements:
                1. The description should explain the main functionality of the tool
                2. Length should be no shorter than {self.min_description_length} characters
                3. Use the same language as the provided parameter descriptions 
                4. Focus on what the tool does, not on specific parameter details
                5. Make it professional and clear"""
            )

            # Extract parameter information for better context
            parameters = api.get("parameters", {})
            param_info = []
            for param_name, param_def in parameters.items():
                param_desc = param_def.get("description", "")
                if param_desc:
                    param_info.append(f"- {param_name}: {param_desc}")
            
            param_context = "\n".join(param_info) if param_info else "No parameter descriptions available"

            user_prompt = textwrap.dedent(
                f"""Tool name: {name}

                Parameter information:
                {param_context}

                Based on the tool name and parameter information above, generate an appropriate description for this tool without any other information:"""
            )

            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=4096
            )
            
            description = answer_text.strip()
            if len(description) >= self.min_description_length:
                return description
                
        except Exception as e:
            logger.error(f"Failed to generate description for {name}: {e}")
        
        return None
    
    def _remove_duplicates(self, apis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate APIs based on name and semantic similarity."""
        if not apis:
            return []
        
        unique_apis = []
        seen_names = set()
        
        for api in tqdm(apis, desc="Removing duplicates"):
            name = api["name"]
            
            # Check for exact name duplicates
            if name in seen_names:
                logger.debug(f"Removed duplicate API: {name}")
                continue
            
            # Check for semantic duplicates using configured similarity method
            is_duplicate = False
            
            # If similarity method is None, skip semantic similarity check
            if not self.similarity_method.is_none():
                for existing_api in unique_apis:
                    if self.similarity_method.is_embedding_based():
                        # Use semantic similarity with embeddings
                        similarity = self._calculate_api_semantical_similarity(api, existing_api)
                    else:
                        # Use text similarity with sequence matcher
                        similarity = self._calculate_text_similarity(
                            f"{api['name']} {api['description']}", 
                            f"{existing_api['name']} {existing_api['description']}"
                        )
                    
                    if similarity > self.duplicate_threshold:
                        logger.debug(f"Removed {self.similarity_method.value} similar API: {name} (similar to {existing_api['name']}, similarity: {similarity:.3f})")
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_apis.append(api)
                seen_names.add(name)
        
        return unique_apis
    
    def _calculate_api_semantical_similarity(self, api1: Dict[str, Any], api2: Dict[str, Any]) -> float:
        """Calculate semantic similarity between two APIs using embeddings."""
        if not self.embeddings or not self.similarity_method.is_embedding_based() or self.similarity_method.is_none():
            return 0.0
        
        try:
            # Create text representations
            text1 = f"{api1['name']} {api1['description']}"
            text2 = f"{api2['name']} {api2['description']}"
            
            # Get embeddings
            embeddings = self.embeddings.embed_texts([text1, text2])
            if len(embeddings) == 2:
                return self.embeddings.cosine_similarity(embeddings[0], embeddings[1])
        except RetryError as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
        except Exception as e:
            logger.error(f"Failed to calculate semantic similarity: {e}")
        
        return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _check_description_clarity(self, api: Dict[str, Any]) -> bool:
        """
        Check if API description is clear and unambiguous using LLM.
        
        Args:
            api: API dictionary to check
            
        Returns:
            True if description is clear, False if needs enhancement
        """
        try:
            system_prompt = textwrap.dedent(
                """You are an API documentation expert. Your task is to evaluate whether a tool specification has clear and unambiguous descriptions.

                ## Evaluation criteria:
                1. **Tool description**: Is it clear what the tool does? (not too vague or confusing)
                2. **Description missing**: Any tool description, parameter description or return description missing?
                3. **Parameter descriptions**: Are parameter purposes clearly explained?
                4. **Return descriptions**: Is the return value clearly described (if present)?
                5. **Overall clarity**: Can a developer understand how to use this tool from the descriptions?

                ## Response format:
                Output a JSON object with the following structure:
                ```json
                {
                    "reason": "Brief explanation of why the specification is clear or unclear",
                    "output": "CLEAR" or "UNCLEAR"
                }
                ```

                ## Examples

                ### Example 1 - Missing Parameter Description
                Input:
                {
                    "name": "delete_user",
                    "description": "The delete_user tool is designed to remove a specified user from the system. This functionality ensures that user data can be effectively managed and maintained according to administrative needs.",
                    "parameters": {
                        "user_id": {
                            "type": "integer"
                        }
                    }
                }
                Reason: The parameter 'user_id' description is missing
                Output: {"reason": "The parameter 'user_id' description is missing", "output": "UNCLEAR"}

                ### Example 2 - Missing Return Description
                Input:
                {
                    "name": "get_user_id_from_name",
                    "description": "Retrieve the user ID based on the provided username.",
                    "parameters": {
                        "username": {
                            "type": "string",
                            "description": "The provided user name"
                        }
                    }
                }
                Reason: The return description is missing
                Output: {"reason": "The return description is missing", "output": "UNCLEAR"}

                ### Example 3 - Vague Tool Description
                Input:
                {
                    "name": "process_data",
                    "description": "Handles data processing operations",
                    "parameters": {
                        "data": {
                            "type": "object",
                            "description": "Input data to process"
                        },
                        "options": {
                            "type": "object", 
                            "description": "Configuration options"
                        }
                    },
                    "returns": {
                        "type": "object",
                        "description": "Processed result"
                    }
                }
                Reason: Tool description is too vague - doesn't specify what kind of processing is done
                Output: {"reason": "Tool description is too vague - doesn't specify what kind of processing is done", "output": "UNCLEAR"}

                ### Example 4 - Clear and Complete
                Input:
                {
                    "name": "calculate_tax",
                    "description": "Calculates income tax based on annual salary and tax bracket information for the specified tax year.",
                    "parameters": {
                        "annual_salary": {
                            "type": "number",
                            "description": "The annual gross salary in USD"
                        },
                        "tax_year": {
                            "type": "integer",
                            "description": "The tax year for which to calculate taxes (e.g., 2023)"
                        },
                        "filing_status": {
                            "type": "string",
                            "description": "Tax filing status: 'single', 'married_joint', or 'married_separate'"
                        }
                    },
                    "returns": {
                        "type": "object",
                        "description": "Tax calculation result containing total_tax, effective_rate, and marginal_rate"
                    }
                }
                Reason: All descriptions are clear, specific, and provide sufficient detail for developers to understand usage
                Output: {"reason": "All descriptions are clear, specific, and provide sufficient detail for developers to understand usage", "output": "CLEAR"}

                Only output the JSON object as specified in the response format."""
            )

            # Format the API for review
            api_text = json.dumps(api, indent=2, ensure_ascii=False)

            user_prompt = textwrap.dedent(
                f"""Please evaluate the clarity of this tool specification:

                {api_text}

                Is this tool specification clear and unambiguous? Answer with only "CLEAR" or "UNCLEAR":"""
            )

            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=100
            )

            # Extract json from response
            response_json = extract_json_from_text(answer_text)

            logger.info(f"Check tool LLM response: {response_json}")

            response = json.loads(response_json)
            response = response.get("output", "UNCLEAR")
            if response == "CLEAR":
                logger.debug(f"API {api['name']}: Description is clear")
                return True
            else:
                logger.debug(f"API {api['name']}: Description is unclear")
                return False
                
        except Exception as e:
            logger.error(f"Failed to check clarity for API {api['name']}: {e}")
            # If check fails, assume enhancement is needed
            return False

    def _enhance_descriptions(self, apis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance API descriptions using LLM for better quality."""
        enhanced_apis = []
        
        for api in tqdm(apis, desc="Enhancing descriptions"):
            try:
                logger.info(f"Processing API: {api['name']}")
                logger.info(f"The original tool description is: {json.dumps(api, indent=2, ensure_ascii=False)}")
                
                # Step 1: Check if description is clear
                is_clear = self._check_description_clarity(api)
                
                if is_clear:
                    # Description is clear, keep original
                    logger.info(f"API {api['name']}: Description is clear, keeping original")
                    enhanced_apis.append(api)
                else:
                    # Description needs enhancement
                    logger.info(f"API {api['name']}: Description needs enhancement")
                    enhanced_api = self._enhance_complete_api(api)
                    enhanced_apis.append(enhanced_api)
                    logger.info(f"The enhanced tool description is: {json.dumps(enhanced_api, indent=2, ensure_ascii=False)}")
                    
            except Exception as e:
                logger.error(f"Failed to process API {api.get('name', 'unknown')}: {e}")
                enhanced_apis.append(api)  # Keep original if processing fails
        
        return enhanced_apis
    
    def _enhance_complete_api(self, api: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance complete API including tool description, parameters, and returns using LLM."""
        try:
            system_prompt = textwrap.dedent(
                """You are an API documentation expert. Improve the provided tool specification by enhancing unclear, ambiguous, or missing descriptions.

                Your task:
                1. Using the same language as the input tool description
                2. Improve the tool description, parameter descriptions, and return descriptions
                3. Output a properly formatted JSON with enhanced descriptions

                Requirements for improvements:
                - Tool description: Clear, concise explanation of main functionality (50-100 characters)
                - Parameter descriptions: Clear explanation of what each parameter does
                - Return descriptions: Clear explanation of what the function returns if original return description is missing or unclear
                - Use professional, technical language
                - Maintain consistency in style and terminology
                - Fix any grammatical errors or unclear phrasing

                Output format: Complete JSON within ```json``` block in the same format as input with improved descriptions"""
            )

            # Format the API for review
            api_text = json.dumps(api, indent=2, ensure_ascii=False)

            user_prompt = textwrap.dedent(
                f"""Please improve this tool specification by enhancing any unclear or ambiguous descriptions:

                {api_text}

                Output the complete improved JSON specification:"""
            )

            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.1),
                max_tokens=4096
            )

            response = extract_json_from_text(answer_text)
            
            # Try to parse the enhanced API
            try:
                enhanced_api = json.loads(response)
                
                # Validate the enhanced API has required structure
                if (isinstance(enhanced_api, dict) and 
                    "name" in enhanced_api and 
                    "description" in enhanced_api and 
                    "parameters" in enhanced_api):
                    
                    logger.debug(f"API {api['name']}: Enhanced successfully")
                    return enhanced_api
                else:
                    logger.warning(f"API {api['name']}: Invalid enhanced format, keeping original")
                    return api
                    
            except json.JSONDecodeError:
                logger.warning(f"API {api['name']}: Failed to parse enhanced response, keeping original")
                return api
                
        except Exception as e:
            logger.error(f"Failed to enhance complete API {api['name']}: {e}")
            return api
    
    def _is_description_clear(self, description: str) -> bool:
        """Check if description is clear and informative."""
        # Simple heuristics for description quality
        if len(description) < 15:
            return False
        if description.count(' ') < 2:  # Too few words
            return False
        if not any(char in description for char in ".,!?"):  # No punctuation
            return False
        return True
    
    def _enhance_single_description(self, api: Dict[str, Any]) -> Optional[str]:
        """Enhance a single API description using LLM."""
        try:
            system_prompt = textwrap.dedent(
                """You are an API documentation expert. Please improve the tool description to make it more clear and informative.

                Requirements:
                1. Keep the original meaning but make the expression clearer
                2. Include the main functionality and purpose of the tool
                3. Length should be approximately 50-100 characters
                4. Use the same language as the current description
                5. Focus on what the tool does based on parameter information
                6. Make it professional and clear"""
            )

            # Extract parameter information for better context
            parameters = api.get("parameters", {})
            param_info = []
            for param_name, param_def in parameters.items():
                param_desc = param_def.get("description", "")
                if param_desc:
                    param_info.append(f"- {param_name}: {param_desc}")
            
            param_context = "\n".join(param_info) if param_info else "No parameter descriptions available"
            
            user_prompt = textwrap.dedent(
                f"""Tool name: {api['name']}
                Current description: {api['description']}
                
                Parameter information:
                {param_context}

                Based on the tool name, current description, and parameter information above, improve this tool's description without any other information:"""
            )

            thinking_content, answer_text, function_calls = generate(
                self.model_config["model"],
                system_prompt,
                user_prompt,
                temperature=self.model_config.get("temperature", 0.3),
                max_tokens=4096
            )
            
            enhanced = answer_text.strip()
            if len(enhanced) >= 20:
                return enhanced
                
        except Exception as e:
            logger.error(f"Failed to enhance description for {api['name']}: {e}")
        
        return None
