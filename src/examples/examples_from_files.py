from strands import Agent
from strands_tools import calculator
from strands_evaluation.dataset import Dataset 
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData
from strands_evaluation.types.evaluation_report import EvaluationReport

### You can only run this file if you have all of the relevant json files from the examples ###

### Recreate/create any custom evaluators ###                                                        
class TestSimilarityEvaluator(Evaluator[str, str]):
    """
    Evaluator that uses ROUGE score to measure similarity between actual and expected outputs.
    
    Calculates ROUGE-1 F-measure and passes the test if the score is greater than 0.5.
    """
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(evaluation_case.actual_output, evaluation_case.expected_output)
        score = scores['rouge1'].fmeasure
        return EvaluationOutput(score = score, test_pass = True if score > .5 else False) 
        
def get_response(query: str) -> str:
    """
    Simple task function that gets a response from an agent for a given query.
    
    Args:
        query: The input query string
        
    Returns:
        The agent's response as a string
    """
    agent = Agent(callback_handler=None)
    return str(agent(query))

def memory_messages_task(inputs: list) -> dict:
    """
    Task function that processes a list of queries sequentially with the same agent,
    preserving conversation context between queries.
    
    Args:
        inputs: List of query strings to process in sequence
        
    Returns:
        Dictionary containing the final response and the full conversation trajectory
    """
    agent = Agent(tools = [calculator], callback_handler=None)
    response = None
    for query in inputs:
        response = agent(query)

    return {
        "output": str(response),
        "trajectory": list(agent.messages)}

def tools_task(query: str) -> dict:
    """
    Task function that uses an agent with calculator tool and tracks tool usage.
    
    Args:
        query: The input query string
        
    Returns:
        Dictionary containing the response and the list of tools used
    """
    agent = Agent(tools = [calculator])
    response = agent(query)
    
    return {"output": str(response),
            "trajectory": list(response.metrics.tool_metrics.keys())}

def from_file_example(file_name: str, task: callable, custom_evaluators: list[Evaluator] = []) -> EvaluationReport:
    """
    Loads a dataset from a JSON file, runs evaluations with the specified task function,
    and returns the evaluation report.
    
    Args:
        file_name: Name of the JSON file (without extension) in the dataset_files directory
        task: Function to execute for each test case in the dataset
        custom_evaluators: List of custom evaluator classes to use for deserialization
        
    Returns:
        Evaluation report with scores and results for all test cases
    """
    dataset = Dataset.from_file(f"./dataset_files/{file_name}.json", "json", custom_evaluators)                       
    report = dataset.run_evaluations(task)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.examples_from_files
    files = ["custom_evaluator_dataset", "judge_output_dataset", "memory_messages_dataset",
             "messages_trajectory_dataset", "tools_trajectory_dataset"] 
    tasks = [get_response, get_response, memory_messages_task, memory_messages_task, tools_task]
    for f, t in zip(files, tasks):
        report = from_file_example(f, t, custom_evaluators=[TestSimilarityEvaluator])
        print(f"File: {f}")
        report.display()
        print("\n")
