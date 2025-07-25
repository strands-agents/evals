
from strands import Agent
from strands_evaluation.dataset import Dataset 
from strands_evaluation.case import Case
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.types.evaluation import EvaluationOutput, EvaluationData

## Using a third party evaluator
from langchain_aws import BedrockLLM
from langchain.evaluation.criteria import CriteriaEvalChain

import asyncio
import datetime

def third_party_example():
    """
    Demonstrates integrating a third-party evaluator (LangChain) with the evaluation framework.
    
    This example:
    1. Creates test cases with expected outputs
    2. Creates a custom evaluator that wraps LangChain's CriteriaEvalChain
    3. Initializes the LangChain evaluator with Bedrock LLM
    4. Creates a dataset with the test cases and evaluator
    5. Saves the dataset to a JSON file
    6. Defines a task function that uses an agent to generate responses
    7. Runs evaluations and returns the report
    
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
    test_case3 = Case(input="When was World War 2?")
    test_case4 = Case(input="Who was the first president of the United States?")

    ### Step 2: Create evaluator using a third party evaluator ###                                                        
    class LangChainCriteriaEvaluator(Evaluator[str, str]):
        def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:

            ## Follow LangChain's Docs: https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.criteria.eval_chain.CriteriaEvalChain.html
            # Initialize Bedrock LLM
            bedrock_llm = BedrockLLM(
                model_id="anthropic.claude-v2",  # or other Bedrock models
                model_kwargs={
                    "max_tokens_to_sample": 256,
                    "temperature": 0.7,
                }
            )

            criteria = {
                "correctness": "Is the actual answer correct?",
                "relevance": "Is the response relevant?"
            }
            
            evaluator = CriteriaEvalChain.from_llm(
                llm=bedrock_llm,
                criteria=criteria
            )
            
            # Pass in required context for evaluator (look at LangChain's docs)
            result = evaluator.evaluate_strings(
                prediction=evaluation_case.actual_output,
                input=evaluation_case.input
            )
            
            # Make sure to return the correct type
            return EvaluationOutput(score=result["score"], test_pass= True if result["score"] > .5 else False, reason = result["reasoning"])
               
    ### Step 3: Create dataset ###                                    
    dataset = Dataset[str, str](cases = [test_case1, test_case2, test_case3, test_case4],
                                evaluator = LangChainCriteriaEvaluator())
    
    ### Step 3.5: Save the dataset ###
    dataset.to_file("third_party_dataset", "json")

    ### Step 4: Define task ###  
    def get_response(query: str) -> str:
        agent = Agent(callback_handler=None)
        return str(agent(query))

    ### Step 5: Run evaluation ###                            
    report = dataset.run_evaluations(get_response)
    return report

async def async_third_party_example():
    """
    Demonstrates integrating a third-party evaluator (LangChain) with the evaluation framework asynchronously.
    
    This example:
    1. Creates test cases with expected outputs
    2. Creates a custom evaluator that wraps LangChain's CriteriaEvalChain
    3. Initializes the LangChain evaluator with Bedrock LLM
    4. Creates a dataset with the test cases and evaluator
    5. Saves the dataset to a JSON file
    6. Defines a task function that uses an agent to generate responses
    7. Runs evaluations and returns the report
    
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
    test_case3 = Case(input="When was World War 2?")
    test_case4 = Case(input="Who was the first president of the United States?")

    ### Step 2: Create evaluator using a third party evaluator ###                                                        
    class LangChainCriteriaEvaluator(Evaluator[str, str]):
        def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:

            ## Follow LangChain's Docs: https://python.langchain.com/api_reference/langchain/evaluation/langchain.evaluation.criteria.eval_chain.CriteriaEvalChain.html
            # Initialize Bedrock LLM
            bedrock_llm = BedrockLLM(
                model_id="anthropic.claude-v2",  # or other Bedrock models
                model_kwargs={
                    "max_tokens_to_sample": 256,
                    "temperature": 0.7,
                }
            )

            criteria = {
                "correctness": "Is the actual answer correct?",
                "relevance": "Is the response relevant?"
            }
            
            evaluator = CriteriaEvalChain.from_llm(
                llm=bedrock_llm,
                criteria=criteria
            )
            
            # Pass in required context for evaluator (look at LangChain's docs)
            result = evaluator.evaluate_strings(
                prediction=evaluation_case.actual_output,
                input=evaluation_case.input
            )
            
            # Make sure to return the correct type
            return EvaluationOutput(score=result["score"], test_pass= True if result["score"] > .5 else False, reason = result["reasoning"])
        
        async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
            return self.evaluate(evaluation_case)
        
    ### Step 3: Create dataset ###                                    
    dataset = Dataset[str, str](cases = [test_case1, test_case2, test_case3, test_case4],
                                evaluator = LangChainCriteriaEvaluator())
    
    ### Step 3.5: Save the dataset ###
    dataset.to_file("async_third_party_dataset", "json")

    ### Step 4: Define task ###  
    async def get_response(query: str) -> str:
        agent = Agent(callback_handler=None)
        response = await agent.invoke_async(query)
        return str(response)

    ### Step 5: Run evaluation ###                            
    report = await dataset.run_evaluations_async(get_response)
    return report

if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.third_party_evaluator 
    start = datetime.datetime.now()
    report = third_party_example()
    end = datetime.datetime.now()
    print("Sync: ", end - start) # Sync:  0:00:33.125273
    report.display(include_input=False)
    report.to_file("third_party_trajectory_report", "json")

    start = datetime.datetime.now()
    report = asyncio.run(async_third_party_example())
    end = datetime.datetime.now()
    print("Async: ", end - start) # Async:  0:00:24.050895
    report.display(include_input=False)
    report.to_file("async_third_party_trajectory_report", "json")
    report.run_display()


    
