"""Agent evaluator for Google ADK evaluation framework."""

import json
import pathlib
import asyncio
import importlib
from typing import Dict, List, Any, Optional

from google.genai import types
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService


class AgentEvaluator:
    """Evaluator for testing agent performance against test cases."""
    
    @staticmethod
    async def evaluate(
        module_name: str,
        test_file_path: str,
        num_runs: int = 1,
        config_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an agent against test cases.
        
        Args:
            module_name: Name of the module containing the agent (e.g., 'machine_learning_engineering')
            test_file_path: Path to the JSON file containing test cases
            num_runs: Number of times to run each test case
            config_file_path: Optional path to configuration file
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load test cases
        test_file = pathlib.Path(test_file_path)
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file_path}")
            
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
            
        # Load config if provided
        config = {}
        if config_file_path:
            config_file = pathlib.Path(config_file_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
        else:
            # Try to find config file in same directory
            config_file = test_file.parent / "test_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
        
        # Import the agent module
        try:
            agent_module = importlib.import_module(f"{module_name}.agent")
            root_agent = agent_module.root_agent
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import agent from {module_name}.agent: {e}")
        
        # Initialize runner
        app_name = module_name.replace("_", "-")
        runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        
        # Run evaluation
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"Running test case {i+1}/{len(test_cases)}: {test_case.get('query', '')[:50]}...")
            
            for run_num in range(num_runs):
                try:
                    result = await AgentEvaluator._run_single_test(
                        runner, test_case, run_num
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error in test case {i+1}, run {run_num+1}: {e}")
                    results.append({
                        'test_case_index': i,
                        'run_number': run_num,
                        'error': str(e),
                        'success': False
                    })
        
        # Calculate metrics
        metrics = AgentEvaluator._calculate_metrics(results, config)
        
        return {
            'test_results': results,
            'metrics': metrics,
            'config': config,
            'num_test_cases': len(test_cases),
            'num_runs': num_runs
        }
    
    @staticmethod
    async def _run_single_test(
        runner: InMemoryRunner,
        test_case: Dict[str, Any],
        run_number: int
    ) -> Dict[str, Any]:
        """Run a single test case."""
        query = test_case.get('query', '')
        expected_tools = test_case.get('expected_tool_use', [])
        expected_responses = test_case.get('expected_intermediate_agent_responses', [])
        reference = test_case.get('reference', '')
        
        # Set up proper workspace state to avoid creating "1" folder in root
        state = runner._ensure_state()
        # Set task_id to a meaningful value for evaluation
        test_task_id = f"eval_test_{run_number}"
        state["task_id"] = test_task_id
        
        # Import config to set up proper workspace directory
        try:
            from machine_learning_engineering.shared_libraries import config
            import os
            # Ensure workspace dir points to the correct location
            workspace_dir = os.path.join(config.CONFIG.workspace_dir, "eval_runs")
            state["workspace_dir"] = workspace_dir
            state["task_name"] = f"test_case_{run_number}"
            state["data_dir"] = config.CONFIG.data_dir
            
            # Create workspace directory structure
            os.makedirs(os.path.join(workspace_dir, state["task_name"], test_task_id), exist_ok=True)
        except ImportError:
            # Fallback if config is not available
            workspace_dir = os.path.join("machine_learning_engineering", "workspace", "eval_runs")
            state["workspace_dir"] = workspace_dir
            state["task_name"] = f"test_case_{run_number}"
        
        # Create session
        session = await runner.session_service.create_session(
            app_name=runner.app_name, 
            user_id=f"test_user_{run_number}"
        )
        
        # Create user message
        content = types.Content(parts=[types.Part(text=query)], role="user")
        
        # Run agent
        response_text = ""
        events = []
        
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            events.append(event)
            if event.content.parts and event.content.parts[0].text:
                response_text = event.content.parts[0].text
        
        # Evaluate response
        response_match_score = AgentEvaluator._calculate_response_match(
            response_text, reference
        )
        
        tool_trajectory_score = AgentEvaluator._calculate_tool_trajectory_score(
            events, expected_tools
        )
        
        return {
            'query': query,
            'response': response_text,
            'reference': reference,
            'response_match_score': response_match_score,
            'tool_trajectory_score': tool_trajectory_score,
            'run_number': run_number,
            'success': True,
            'events_count': len(events)
        }
    
    @staticmethod
    def _calculate_response_match(response: str, reference: str) -> float:
        """Calculate similarity between response and reference."""
        if not reference:
            return 1.0  # If no reference, consider it a pass
            
        if not response:
            return 0.0
            
        # Simple keyword-based matching
        response_lower = response.lower()
        reference_lower = reference.lower()
        
        # Check if key concepts from reference appear in response
        reference_words = set(reference_lower.split())
        response_words = set(response_lower.split())
        
        if not reference_words:
            return 1.0
            
        # Calculate overlap ratio
        overlap = len(reference_words.intersection(response_words))
        return overlap / len(reference_words)
    
    @staticmethod
    def _calculate_tool_trajectory_score(events: List[Any], expected_tools: List[str]) -> float:
        """Calculate tool usage trajectory score."""
        # For now, return 1.0 if no tools expected, 0.5 otherwise
        # This is a placeholder implementation
        if not expected_tools:
            return 1.0
        return 0.5
    
    @staticmethod
    def _calculate_metrics(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics from results."""
        if not results:
            return {}
            
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {
                'overall_score': 0.0,
                'success_rate': 0.0,
                'response_match_avg': 0.0,
                'tool_trajectory_avg': 0.0
            }
        
        # Calculate averages
        response_scores = [r.get('response_match_score', 0.0) for r in successful_results]
        tool_scores = [r.get('tool_trajectory_score', 0.0) for r in successful_results]
        
        response_match_avg = sum(response_scores) / len(response_scores) if response_scores else 0.0
        tool_trajectory_avg = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0
        
        # Get weights from config
        criteria = config.get('criteria', {})
        response_weight = criteria.get('response_match_score', 0.5)
        tool_weight = criteria.get('tool_trajectory_avg_score', 0.5)
        
        # Calculate weighted overall score
        overall_score = (response_match_avg * response_weight + 
                        tool_trajectory_avg * tool_weight)
        
        success_rate = len(successful_results) / len(results)
        
        return {
            'overall_score': overall_score,
            'success_rate': success_rate,
            'response_match_avg': response_match_avg,
            'tool_trajectory_avg': tool_trajectory_avg,
            'total_tests': len(results),
            'successful_tests': len(successful_results)
        }
