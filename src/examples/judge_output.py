from strands import Agent
from strands_evaluation.dataset import Dataset
from strands_evaluation.case import Case
from strands_evaluation.evaluators.output_evaluator import OutputEvaluator
import asyncio
import datetime

def output_judge_example():
   """
   Demonstrates using an LLM-based evaluator to judge agent outputs.
   
   This example:
   1. Creates test cases
   2. Creates an OutputEvaluator with a specified rubric
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
                                 expected_output="Water is clear.",
                                 metadata={"category": "knowledge"})
   
   test_case3 = Case(input="When was World War 2?")
   test_case4 = Case(input="Who was the first president of the United States?")

   ### Step 2: Create evaluator ###   
   LLM_judge = OutputEvaluator(rubric="The output should represent a reasonable answer to the input.",
                           include_inputs = True)
   ## or 
   LLM_judge_w_prompt = OutputEvaluator(rubric="The output should represent a reasonable answer to the input.",
                                 system_prompt = "You are an expert AI evaluator. Your job is to assess the quality of the response based according to a user-specified rubric. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}",
                                 include_inputs = True)

   ### Step 3: Create dataset ###                                                                
   dataset = Dataset[str, str](cases = [test_case1, test_case2, test_case3, test_case4],
                                             evaluator = LLM_judge)
   ### Step 3.5: Save the dataset ###
   dataset.to_file("judge_output_dataset", "json")

   ### Step 4: Define task ###                                      
   # simple example here but could be more complex depending on the user's needs
   def get_response(query: str) -> str:
      agent = Agent(callback_handler=None)
      return str(agent(query))

   ### Step 5: Run evaluation ###                                                                                                                                                
   report = dataset.run_evaluations(get_response)
   return report

async def async_output_judge_example():
   """
   Demonstrates using an LLM-based evaluator to judge agent outputs asynchronously.
   
   This example:
   1. Creates test cases
   2. Creates an OutputEvaluator with a specified rubric
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
                                 metadata={"category": "knowledge"})
   
   test_case3 = Case(input="When was World War 2?")
   test_case4 = Case(input="Who was the first president of the United States?")

   ### Step 2: Create evaluator ###   
   LLM_judge = OutputEvaluator(rubric="The output should represent a reasonable answer to the input.",
                           include_inputs = True)
   ## or 
   LLM_judge_w_prompt = OutputEvaluator(rubric="The output should represent a reasonable answer to the input.",
                                 system_prompt = "You are an expert AI evaluator. Your job is to assess the quality of the response based according to a user-specified rubric. You respond with a JSON object with this structure: {reason: string, pass: boolean, score: number}",
                                 include_inputs = True)

   ### Step 3: Create dataset ###                                                                
   dataset = Dataset[str, str](cases = [test_case1, test_case2, test_case3, test_case4],
                                             evaluator = LLM_judge)
   ### Step 3.5: Save the dataset ###
   dataset.to_file("async_judge_output_dataset", "json")

   ### Step 4: Define task ###                                      
   # simple example here but could be more complex depending on the user's needs
   async def get_response(query: str) -> str:
      agent = Agent(callback_handler=None)
      response = await agent.invoke_async(query)
      return str(response)

   ### Step 5: Run evaluation ###                                                                                                                                                
   report = await dataset.run_evaluations_async(get_response)
   return report

if __name__ == "__main__":
   # run the file as a module: eg. python -m examples.judge_output 
   start_time = datetime.datetime.now()
   report = output_judge_example()
   end_time = datetime.datetime.now()
   print("Sync: ", end_time - start_time) # Sync:  0:00:29.743616
   report.display()
   report.to_file("judge_output_report", "json")

   start_time = datetime.datetime.now()
   report = asyncio.run(async_output_judge_example())
   end_time = datetime.datetime.now()
   print("Async: ", end_time - start_time)  # Async:  0:00:10.716829
   # report.display()
   report.to_file("async_judge_output_report", "json")
   report.run_display()