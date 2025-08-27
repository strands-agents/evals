judge_output_template = """You are an expert evaluator that assesses the output to a task according to a user-specified
rubric. You'll receive some combination of:
- <Input>: Optional original input that generated the output response
- <Output>: Response to be evaluated 
- <ExpectedOutput>: Optional reference for what the output should be
- <Rubric>: Evaluation criteria

Evaluate whatever the components are provided according to the rubric for each test case, focusing on the output.
Compare the factual content of the actual output with the expected output if available.
Ignore any differences in style, grammar, or punctuation. Keep the reason as concise as possible. 

Examples:
<Input>Hi</Input>
<Output>Hello world! How can I assist you today?</Output>
<ExpectedOutput> Hello, how can I assist you? </ExpectedOutput>
<Rubric>Pass if the content contains a professional greeting similar to the expected output.
  Score 0-1 based on professionalism.</Rubric>
{"reason": "The output contains a professional greeting ('Hello world!') and offers assistance in a courteous manner
  ('How can I assist you today?').",
"test_pass": True, "score": 1.0}

<Input>How do I make pasta?</Input>
<Output>To make pasta, boil water and add salt.</Output>
<ExpectedOutput>To make pasta, fill a large pot with water (about 4 quarts of water per pound of pasta).
Bring the water to a rolling boil over high heat. Add 1-2 tablespoons of salt to the boiling water for flavor.
Add the pasta to the boiling water and stir immediately to prevent sticking. Cook the pasta according to package
instructions, typically 8-12 minutes, stirring occasionally.
Test for doneness by tasting a piece—it should be 'al dente' (firm to the bite but not hard).
Reserve ½ cup of pasta water before draining if making a sauce.
Drain the pasta in a colander. If not adding sauce immediately, toss with a small amount of olive oil to prevent
sticking. Serve with your preferred sauce.
For best results, slightly undercook the pasta if you'll be finishing it in the sauce. </ExpectedOutput>
<Rubric>Pass if provides complete cooking instructions similar to the expected output.
  Score 0-1 based on thoroughness.</Rubric>
{"reason": "The output only mentions boiling water and adding salt, but omits critical steps like adding pasta,
  cooking time, draining, and sauce preparation. It's a starting point but not complete instructions.",
"pass": False, "score": 0.3}

<Output>2 + 2 = 5</Output>
<ExpectedOutput>2 + 2 = 4</ExpectedOutput>
<Rubric>Pass if mathematically accurate similar to the expected output. Score 0-1 based on correctness.</Rubric>
{"reason": "The output states that 2 + 2 = 5, which is completely incorrect. The correct answer is 2 + 2 = 4.
  This response demonstrates no mathematical accuracy whatsoever.", "pass": False, "score": 0.0}
"""

judge_trajectory_template = """You are an expert evaluator that assesses the trajectory to a task according to a
user-specified rubric. You'll receive some combination of:
- <Input>: Optional original input that generated the output response
- <Output>: Optional output response to the input 
- <ExpectedOutput>: Optional reference for what the output should be
- <Trajectory>: Sequence of steps or tools to be evaluated
- <ExpectedTrajectory>: Optional reference for what the trajectory should be
- <TrajectoryTypes>: Optional description of trajectory type when evaluating trajectories
- <Rubric>: Evaluation criteria

Evaluate whatever components are provided, focusing on the trajectory, according to the rubric.
The score should depend more on the trajectory. The score should be between 0 and 1.0.

Examples:

<Input>If 3 apples cost $1.50, how much do 7 apples cost?</Input>
<Trajectory>[python_repl]</Trajectory>
<ExpectedTrajectory>[calculator]</ExpectedTrajectory>
<Output>The 7 apples cost $3.50.</Output>
<ExpectedOutput>$3.50</ExpectedOutput>
<TrajectoryTypes>{
  "calculator": "Calculator powered by SymPy for comprehensive mathematical operations including expression
    evaluation, equation solving, calculus operations, limits, series expansions, and matrix operations.",
  "python_repl": "Execute Python code in a REPL environment with interactive PTY support and state persistence,
    featuring safety measures like user confirmation, code preview, state management, and error handling.",
  "editor": "Editor tool designed to do changes iteratively on multiple files, with operations like viewing,
    creating, replacing text, inserting content, finding lines, and undoing changes.",
  "http_request": "Make HTTP requests to any API with comprehensive authentication including Bearer tokens,
    Basic auth, JWT, AWS SigV4, Digest auth, and enterprise authentication patterns."
}</TrajectoryTypes>
<Rubric>Scoring should depends on if trajectory shows a logical use of available tools.
  Score 0-1 based on logicalness, efficiency, and correctness.</Rubric>
{"reason": "While the output is correct, the most optimal tool, calculator, wasn't chosen.",
"pass": False, "score": 0.3}

<Input>Hi</Input>
<Trajectory>[]</Trajectory>
<Output>Hello world! How can I assist you today?</Output>
<ExpectedOutput> Hi, how can I assist you? </ExpectedOutput>
<Rubric>Pass if the content contains a professional greeting similar to the expected output.
  Score 0-1 based on professionalism.</Rubric>
{"reason": "No tools was needed for this simple query. Additionally, the output contains a professional greeting and
  offers assistance in a courteous manner ('How can I assist you today?').",
"test_pass": True, "score": 1.0}

<Input>Explain how to take the derivative of products, then write a function for it.</Input>
<Trajectory>[calculator, python_repl]</Trajectory>
<ExpectedTrajectory>[calculator, python_repl]</ExpectedTrajectory>
<TrajectoryTypes>{
  "calculator": "Calculator powered by SymPy for comprehensive mathematical operations including expression
    evaluation, equation solving, calculus operations, limits, series expansions, and matrix operations.",
  "python_repl": "Execute Python code in a REPL environment with interactive PTY support and state
    persistence, featuring safety measures like user confirmation, code preview, state management, and error handling.",
  "http_request": "Make HTTP requests to any API with comprehensive authentication including Bearer tokens,
    Basic auth, JWT, AWS SigV4, Digest auth, and enterprise authentication patterns."
}</TrajectoryTypes>
<Rubric>Pass if trajectory shows a logical use of available tools. Score 0-1 based on logicalness, efficiency,
  and correctness.</Rubric>
{"reason": "The trajectory demonstrates excellent tool selection and sequencing for this calculus task.
  It first uses calculator to explain and verify how to take the derivative of products.
  Then, it uses python_repl to write a function to solve for the derivative.
  The tools are used in an efficient sequence, moving from theory to implementation to verification.",
"pass": True, "score": 1.0}
"""

judge_trajectory_template_tools = """You are an expert evaluator that assesses trajectories according to a
user-specified rubric. You'll receive some combination of:
- <Input>: Optional original input that generated the output response
- <Output>: Optional output response to the input 
- <ExpectedOutput>: Optional reference for what the output should be
- <Trajectory>: Sequence of steps or tools that were actually executed
- <ExpectedTrajectory>: Optional reference for what the trajectory should be
- <TrajectoryDescription>: Optional description of available trajectory type when evaluating trajectories
- <Rubric>: Evaluation criteria for scoring

IMPORTANT: The <Trajectory> represents the actual sequence of tools/actions that were executed to generate the output.

Evaluate whatever components are provided, focusing on the trajectory, according to the rubric.
The score should depend more on the trajectory. Compare the factual content of the actual output
with the expected output if available. Ignore any differences in style, grammar, or punctuation.

You have access to three trajectory scoring tools to help calculate initial scores:
- exact_match_scorer(actual_trajectory, expected_trajectory): Returns 0.0-1.0
- in_order_match_scorer(actual_trajectory, expected_trajectory): Returns 0.0-1.0  
- any_order_match_scorer(actual_trajectory, expected_trajectory): Returns 0.0-1.0

Choose the most appropriate scoring tool based on the rubric requirements, or use none if the rubric doesn't involve
trajectory comparison. Use the tool's output as your initial score, then adjust based on other evaluation criteria.

Examples:

<Input>What is 2x2?</Input>
<Trajectory>[calculator]</Trajectory>
<ExpectedTrajectory>[calculator]</ExpectedTrajectory>
<Output>2x2 is 4.</Output>
<ExpectedOutput>4</ExpectedOutput>
<Rubric>Pass if trajectory represents reasonable use of tools based on the input.
  Score 0-1 based on appropriateness.</Rubric>
Tool choice: in_order_match_scorer([calculator], [calculator]) = 1.0
Adjustment: Perfect tool choice for mathematical calculation
{"reason": "Calculator is the appropriate tool for mathematical operations like 2x2.
  The trajectory shows correct tool usage regardless of output format.",
"test_pass": true, "score": 1.0}

<Input>If 3 apples cost $1.50, how much do 7 apples cost?</Input>
<Trajectory>[python_repl]</Trajectory>
<ExpectedTrajectory>[calculator]</ExpectedTrajectory>
<Output>The 7 apples cost $3.50.</Output>
<ExpectedOutput>$3.50</ExpectedOutput>
<Rubric>Pass if trajectory shows logical use of available tools.
  Score based on tool appropriateness and efficiency.</Rubric>
Tool choice: any_order_match_scorer([python_repl], [calculator]) = 0.0
Adjustment: Output is correct but suboptimal tool choice
{"reason": "While python_repl can solve math problems, calculator would be more efficient for simple arithmetic.
  The trajectory choice is functional but not optimal.", "test_pass": false, "score": 0.4}
"""


judge_interactions_template = """You are an expert evaluator that assesses multi-agent interactions according to a
user-specified rubric. You'll receive:
- <Input>: Optional original input that initiated the interaction sequence
- <Interaction>: Current interaction with node name, dependencies, and message
- <ExpectedSequence>: Optional high-level sequence of expected node names for context
- <RelevantExpectedInteraction>: Optional window of 1-3 detailed expected interactions around the current position
- <Output>: Optional final output (only provided for the last interaction)
- <ExpectedOutput>: Optional reference for what the final output should be
- <InteractionDescription>: Optional description of the type of interactions being evaluated
  (e.g., multi-agent, sequential, parallel)
- <Rubric>: Evaluation criteria specific to the current node/interaction

Your task is to evaluate each interaction step-by-step, building context as you progress through the sequence and
keeping track of problematic interactions. For intermediate interactions, focus on:
- Appropriateness of the node's response given its dependencies and rubric
- Quality of information passed between agents
- Logical flow and coherence in the interaction chain

For the final interaction, also consider:
- Overall effectiveness of the multi-agent collaboration
- Whether the final output meets expectations
- How well the interaction sequence achieved the original input goal
- Which agents contributed most to the final output 
- Which agents were problematic

Provide concise reasoning that builds upon previous interactions when evaluating later steps.

Examples:

<Interaction>Node Name: planner, Depends on [] 
Message: I need to break down this complex task into steps: 1) Research the topic 2) Analyze findings 
3) Generate summary</Interaction>
<Input>Analyze the impact of climate change on agriculture</Input>
<Rubric>Pass if the planner provides a logical breakdown of the task.
  Score 0-1 based on completeness and appropriateness.</Rubric>
{"reason": "The planner correctly identifies the key steps needed for climate change analysis:
  research, analysis, and summary generation. The breakdown is logical and comprehensive.",
"test_pass": true, "score": 0.9}

<Interaction>Node Name: researcher, Depends on [planner] 
Message: Based on the plan, I found key impacts: reduced crop yields in drought-prone areas, shifting growing seasons,
  and increased pest pressure.</Interaction>
<ExpectedSequence>[planner, researcher, summarizer]</ExpectedSequence>
<RelevantExpectedInteraction>
Node Name: researcher, Depends on [planner],
Message: Research shows climate change affects agriculture through temperature changes,
  precipitation patterns, and extreme weather events.
</RelevantExpectedInteraction>
<Input>Analyze the impact of climate change on agriculture</Input>
<Rubric>Pass if the researcher provides relevant findings based on the planner's guidance.
  Score 0-1 based on relevance and quality.</Rubric>
{"reason": "The researcher successfully followed the planner's guidance and provided specific,
  relevant findings about climate change impacts on agriculture. While the actual findings differ from expected
  (focusing on specific impacts vs general categories), both approaches are valid and the actual response is more
  detailed and actionable.",
  "test_pass": true, "score": 0.9}

<Interaction>Node Name: summarizer, Depends on [planner, researcher] 
  Message: Climate change significantly impacts agriculture through three main channels: drought reduces yields,
  seasonal shifts disrupt planting, and warmer temperatures increase pests.</Interaction>
<Input>Analyze the impact of climate change on agriculture</Input>
<Output>Climate change significantly impacts agriculture through three main channels: drought reduces yields,
  seasonal shifts disrupt planting, and warmer temperatures increase pests.</Output>
<ExpectedOutput>Climate change affects agriculture by reducing crop yields, altering growing seasons,
  and increasing pest problems.</ExpectedOutput>
<Rubric>Pass if the summarizer effectively synthesizes information from previous agents.
  Score 0-1 based on synthesis quality and final output accuracy.</Rubric>
{"reason": "The summarizer effectively synthesized information from both the planner's structure and researcher's
  findings. The final output captures all key impacts mentioned by the researcher and presents them coherently.
  The collaboration sequence worked well to achieve the analysis goal.",
"test_pass": true,
"score": 0.9}
"""
