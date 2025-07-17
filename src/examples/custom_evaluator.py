from strands import Agent

from strands_evaluation.dataset import Dataset 
from strands_evaluation.case import Case
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData

def simple_example():
    """
    Demonstrates creating and using a custom evaluator with ROUGE score metrics.
    
    This example:
    1. Creates test cases with expected outputs
    2. Defines a custom evaluator using ROUGE similarity scoring
    3. Creates a dataset with the test cases and evaluator
    4. Saves the dataset to a JSON file
    5. Defines a task function that uses an agent to generate responses
    6. Runs evaluations and returns the report
    
    Returns:
        EvaluationReport: The evaluation results
    """
    ### Step 1: Create test cases ###
    test_case1 = Case[str, str](name = "knowledge-1",
                                    input="What is the capital of France?",
                                    expected_output="The capital of France is Paris.",
                                    metadata={"category": "knowledge"})
                                    
    test_case2 = Case[str, str](name = "knowledge-2",
                                    input="What color is the ocean?",
                                    expected_output="The ocean is blue.",
                                    metadata={"category": "knowledge"})

    ### Step 2: Create evaluator ###                                                        
    class TestSimilarityEvaluator(Evaluator[str, str]):
        def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(evaluation_case.actual_output, evaluation_case.expected_output)
            score = scores['rouge1'].fmeasure
            return EvaluationOutput(score = score, test_pass = True if score > .5 else False)        
            
    ### Step 3: Create dataset ###                                    
    dataset = Dataset[str, str](cases = [test_case1, test_case2],
                                evaluator = TestSimilarityEvaluator())
    ### Step 3.5: Save the dataset ###
    dataset.to_file("custom_evaluator_dataset", "json")
    
    ### Step 4: Define task ###  
    def get_response(query: str) -> str:
        agent = Agent(callback_handler=None)
        return str(agent(query))

    ### Step 5: Run evaluation ###                            
    report = dataset.run_evaluations(get_response)
    return report


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.custom_evaluator 
    report = simple_example()
    print(report)
    report.to_file("custom_report", "json")
    report.display(include_input=False)