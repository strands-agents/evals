"""Failure detection prompt template v0.

Ported verbatim from AgentCoreLens failure_detection.j2.
Only the rendering mechanism changed (Jinja2 -> Python f-strings).
"""

SYSTEM_PROMPT = """\
# LLM Annotation Prompt for Agent Session Data

## System Role and Task Overview

You are an expert AI systems evaluator tasked with annotating failure patterns \
in AI agent session data. Your goal is to identify and categorize errors that \
occur during agent interactions, following a systematic approach to failure \
detection and classification.

### Task Overview

You will be provided with a session containing a list of traces. Each trace \
represents a complete interaction cycle and contains spans representing atomic \
operations (agent invocations, tool calls, LLM inference). Your task is to:

1. **Analyze each span** in the session for potential failures
2. **Categorize failures** using the predefined taxonomy
3. **Skip spans** where no errors are identified
4. **Output annotations** in the specified JSON format"""

DEFAULT_ANNOTATION_GUIDELINES = """\
### Core Principles
- **Be consistent**: Apply the same standards across all sessions
- **Be critical**: Don't assume the agent's actions are correct
- **Be precise**: Use the most specific category available
- **Be exhaustive**: Record every distinct failure you observe
- **Be objective**: Focus on observable evidence, not inferred causes
- **Use "other"**: If uncertain about category, use "other" with explanation

### Special Attention: execution-error
**Don't overlook obvious execution failures.** While maintaining thorough \
coverage of all error categories, be particularly vigilant for explicit \
"execution-error-*" category failures across all `span_type`s, as these \
manifested failures can sometimes be dismissed despite being clearly \
observable in system outputs and feedback.

### Span-by-Span Analysis Process

1. **Read the complete session** to understand context, goals, and agent behavior
2. **For each span** (within each trace):
   - If no failure detected \u2192 Skip this span (do not annotate)
   - If failure detected \u2192 Assign appropriate category and provide evidence"""

CATEGORIES = """\
Use the specific category names (not parent categories) in your annotations. \
The details for each category are described in the following table organized \
by parent categories:

|Parent Category|Category|Category Definition|Examples|
|---|---|---|---|
|**execution-error**|execution-error-category-authentication|Failed access attempts due to credential or permission issues\
<br><br> \u2022 HTTP 401/403 errors<br> \u2022 Invalid/expired tokens<br> \u2022 Missing credentials|\
\u2022 "Invalid API key provided"<br> \u2022 "Access token expired"<br> \u2022 "Unauthorized access to Spotify API"|
||execution-error-category-resource-not-found|Requested resource does not exist at specified location\
<br><br> \u2022 HTTP 404 errors<br> \u2022 Missing file/endpoint<br> \u2022 Invalid identifier|\
\u2022 "Playlist ID not found"<br> "Endpoint /api/v2/users does not exist"<br> \u2022 "Image file missing from specified path"|
||execution-error-category-service-errors|Upstream service failure or unavailability\
<br><br> \u2022 HTTP 500 errors<br> \u2022 Service disruption<br> \u2022 System failures|\
\u2022 "Internal server error in database connection"<br> \u2022 "AWS service temporarily unavailable"<br> \
"Google API service disruption"|
||execution-error-category-rate-limiting|Request frequency exceeds allowed quota\
<br><br> \u2022 HTTP 429 errors<br> \u2022 Quota exceeded messages<br> \u2022 Rate threshold warnings|\
\u2022 "Too many requests to Twitter API"<br> \u2022 "Rate limit exceeded: try again in 60 seconds"<br> \
\u2022 "Daily quota exceeded for Google Maps API"|
||execution-error-category-formatting|Failure to produce correctly structured output according to expected \
format or syntax<br><br> \u2022 Syntax errors in structured data<br> \u2022 Missing required structural elements<br> \
\u2022 Invalid format for specified type|\u2022 JSON missing closing bracket<br> \u2022 SQL query with incorrect \
syntax<br> \u2022 HTML with unclosed tags|
||execution-error-category-timeout|Request or operation exceeded time constraints\
<br><br> \u2022 Explicit timeout message<br> \u2022 Duration threshold exceeded<br> \u2022 Connection time limit|\
\u2022 "Request timed out after 30 seconds"<br> \u2022 "Function execution exceeded time limit"<br> \
\u2022 "Connection timeout to external API"|
||execution-error-category-resource-exhaustion|System resource capacity limits reached\
<br><br> \u2022 Memory/disk/CPU limits<br> \u2022 Resource quota exceeded<br> \u2022 System capacity error|\
\u2022 "Out of memory error during image processing"<br> \u2022 "Disk space full during file download"<br> \
\u2022 "CPU quota exceeded"|
||execution-error-category-environment|Missing or incorrect system setup requirements\
<br><br> \u2022 Missing environment variables<br> \u2022 Invalid configuration<br> \u2022 Setup requirement errors|\
\u2022 "OPENAI_API_KEY not found in environment"<br> \u2022 "Missing write permissions for database"<br> \
\u2022 "Required configuration file not present"|
||execution-error-category-tool-schema|Tool input/output does not match required schema\
<br><br> \u2022 Parameter validation errors<br> \u2022 Missing required fields<br> \u2022 Invalid data types|\
\u2022 "Expected parameter 'date' to be ISO format"<br> \u2022 "Required field 'user_id' missing"<br> \
\u2022 "Invalid enum value for 'sort_order'"|
|**task-instruction**|task-instruction-category-non-compliance|Failure to follow explicit directives from the \
user or system prompts/constraints<br><br>Not all datasets have this information|System: "Always authenticate \
user before providing account information"<br>User: "What's my account balance?"<br>Agent: *immediately provides \
balance without authentication*|
||task-instruction-category-problem-id|Agent correctly identifies the end goal but fails to determine the \
appropriate path to achieve it. The agent applies incorrect reasoning about how to reach the goal, leading to \
misaligned actions.|Goal: Fix SQL indentation issue<br>User: "There's an extra space appearing in WITH clause \
indentation in L003.py"<br><br>Agent: *Opens and analyzes core/parser/segments.py*<br>"I'll help fix the \
indentation issue. Looking at segments.py which handles SQL parsing..."<br>[Proceeds to analyze general parsing \
logic]<br><br>[\\u2713 Goal: Fix indentation spacing issue<br> \\u2717 Path: Wrong file - should examine L003.py \
which handles indentation rules<br> \\u2717 Reasoning: Agent incorrectly assumes indentation is handled in general \
parser rather than specific rule file]|
|**incorrect-actions**|incorrect-actions-category-tool-selection|Using an incorrect but available tool when a \
more appropriate tool exists in the agent's toolbox<br><br>\u2022 Tool exists in toolbox<br> \u2022 Wrong tool \
selected for task<br> \u2022 More appropriate tool available<br> \u2022 Not about hallucinated tools|\
\u2022 Using search_flights() when check_flight_status() exists and is appropriate<br> \u2022 Using \
general_search() when specific domain_search() available<br> \u2022 Using text_analysis() when \
numerical_calculator() needed|
||incorrect-actions-category-poor-information-retrieval|Incorrect or ineffective use of a correctly selected \
tool<br><br>\u2022 Right tool, wrong query<br> \u2022 Irrelevant search terms<br> \u2022 Missing key search \
parameters<br> \u2022 Overly broad/narrow queries|\u2022 Searching "airplane" when looking for specific flight \
status<br> \u2022 Using too generic terms in domain-specific search<br> \u2022 Missing critical search parameters|
||incorrect-actions-category-clarification|Proceeding with action despite unclear/incomplete information\
<br><br>\u2022 Missing critical info<br> \u2022 Ambiguous user input<br> \u2022 Assumptions made instead of asking|\
\u2022 Booking flight without confirming date<br> \u2022 Processing refund without asking reason<br> \
\u2022 Modifying order without confirming which one|
||incorrect-actions-category-inappropriate-info-request|Agent makes an information request that is inappropriate \
for at least one of these reasons:<br><br>1. Tool-Available: Information should be retrieved through agent's \
available tools/APIs<br>-> Is there an API/tool that should be used instead?<br><br>2.. Task-Irrelevant: \
Information is not needed for any stage of the current task<br>-> Is the information unnecessary for the task?\
<br><br><br>Differences compared to other categories:<br>\u2022 First-time request (if repeated, see Information \
Seeking Repetition)<br>|\u2022 Agent to user: "What's the current exchange rate?" (should use currency_api \
instead)<br>\u2022 Agent to user: "What's your passport number?" (when just browsing flights, not booking)<br>\
\u2022  Agent to user: "What's the weather like in Paris?" (when agent has weather_api access)|
|**context-handling-error**|context-handling-error-category-context-handling-failures|Sudden and unwarranted \
loss of recent interaction context while retaining older context, or complete and unexpected restart of \
conversation state, losing all accumulated context|\u2022 Agent forgets user's last selection but remembers \
initial request<br> \u2022 Agent asks for information provided in last 2-3 turns<br> \u2022 Agent reverts to \
earlier conversation state<br> \u2022 Agent suddenly greets user as new in middle of booking<br> \u2022 Agent \
loses all progress in multi-step process<br> \u2022 Agent restarts task from beginning|
|**hallucination**|hallucination-category-hall-capabilities|Agent claims or attempts to use tool features that \
don't exist in the API specification|\u2022 "I'll use search_flights.filter_by_meal_preference()"<br> \u2022 \
"The email_tool can automatically translate messages"<br> \u2022 "Let me use the AI image generator" (when no \
such tool exists)|
||hallucination-category-hall-misunderstand|Agent misinterprets the meaning or structure of actual tool responses \
while the response itself is valid.|\u2022 Interpreting error code as success<br> \u2022 Treating warning as \
confirmation<br> \u2022 Misreading data structure|
||hallucination-category-hall-usage|Agent claims to have used tools when it hasn't|\u2022 "I've checked the \
database" (without tool call)<br> \u2022 "I've verified with the API" (no verification done)<br> \u2022 "The \
tool confirmed..." (no tool used)|
||hallucination-category-hall-history|Agent references conversations or user inputs that never occurred|\u2022 \
"As you mentioned earlier..." (when not mentioned)<br> \u2022 "Based on your preference for..." (never stated)\
<br> \u2022 "Following up on your request..." (no such request)|
||hallucination-category-hall-params|Agent uses parameter values that conflict with established previous \
trajectory context|\u2022 User: "I want to fly to Paris" <br>  [later] Agent: "Where would you like to fly to?"\
<br>\u2022 User: "My account number is [REDACTED:BANK_ACCOUNT_NUMBER]" <br>  [later] Agent: "Can you provide \
your account number?"<br>\u2022 User: "I prefer morning flights" <br>  [later] Agent: "Do you have a preference \
for flight time?"|
||hallucination-category-fabricate-tool-outputs|Agent fabricates or makes up tool outputs that were not actually \
returned|\u2022 Agent claims API returned data that was never received<br> \u2022 Agent invents success messages\
<br> \u2022 Agent creates fake error responses|
|**repetitive-behavior**|repetitive-behavior-category-repetition-tool|Making identical API/tool calls multiple \
times without justification<br><br> \u2022 Same tool, same parameters <br>\u2022 No user request ("provide me \
with 2 more options") nor justification (the result of "check account balance" would be different before and \
after making a withdrawal, so calling it twice could be justified)|\u2022 Calling check_flight_status(\
flight_id="123") repeatedly<br> \u2022 Multiple identical search queries<br> \u2022 Repeating API \
authentication calls|
||repetitive-behavior-category-repetition-info|Requesting information from user that was previously provided in \
the conversation<br><br> \u2022 Asking for provided data<br> \u2022 Ignoring prior responses<br> \u2022 \
Redundant user queries<br> \u2022 Forgetting stated preferences<br><br>The key distinction from Inappropriate \
Information Request is that the information being requested:<br>- Would be appropriate if asked for the first \
time<br>- Cannot be obtained through tools/APIs<br>- Is relevant to the current task<br>- Was already directly \
stated by the user in the conversation|\u2022 "What's your destination?" (after already asked)<br> \u2022 \
"Please provide your user ID" (already given)<br> \u2022 "What's your preferred date?" (previously stated)|
||repetitive-behavior-category-step-repetition|Repeating the same action steps or workflow stages without \
progress or justification|\u2022 Agent repeats same verification step multiple times<br> \u2022 Cycling through \
same process without advancing<br> \u2022 Re-executing completed steps|
|**orchestration-related-errors**|orchestration-related-errors-category-reasoning-mismatch|Disconnect between \
agent's stated reasoning/plan and actual executed actions|\u2022 Planning to verify but skipping verification\
<br> \u2022 Stating one approach but executing another<br> \u2022 Reasoning about safety but ignoring checks|
||orchestration-related-errors-category-goal-deviation|Agent diverges from original task objective or user \
intent|\u2022 Booking flight becomes travel planning<br> \u2022 Simple query becomes complex analysis<br> \
\u2022 Support request becomes sales opportunity|
||orchestration-related-errors-category-premature-termination|Task or interaction ended before completion of \
necessary steps or objectives|\u2022 Ending before payment confirmation<br> \u2022 Missing final verifications\
<br> \u2022 Incomplete data collection<br> \u2022 Skipped error handling|
||orchestration-related-errors-category-unaware-termination|Failure to recognize or properly handle task \
completion or continuation criteria|\u2022 Continuing after task completion<br> \u2022 Missing natural end \
points<br> \u2022 Ignoring completion signals<br> \u2022 Redundant actions|
|**llm-output**|llm-output-category-nonsensical|Exposing internal system details, implementation logic, or \
debug information in user-facing responses; producing malformed, illogical, or endlessly looping tool calls or \
JSON structures; or displaying redacted, placeholder, or special tokens|\u2022 "As per my system prompt..."<br> \
\u2022 "According to my training..."<br> \u2022 "Error in function call_api() line 234"<br> \u2022 "Internal \
state: processing_step_2"<br> \u2022 Showing API keys/endpoints<br> \u2022 {"tool": {"tool": {"tool": \
{...}}}}<br> \u2022 Mixing user text and JSON<br> \u2022 Invalid tool call sequences<br> \u2022 "Hello \
[MASK], how are you?"<br> \u2022 "Your balance is <VALUE_TOKEN>"|
|**configuration-mismatch**|configuration-mismatch-category-tool-definition|Tool functionality differs from its \
declared purpose or capabilities<br>Disconnect between API requirements and agent's actual capabilities<br><br>\
The root cause of the issue is in the set up|\u2022 Calculator tool declared as search engine<br> \u2022 Text \
processor declared as numerical tool<br> \u2022 Simple lookup declared as complex analysis<br><br>-----------\
<br><br>\u2022 POST body requirements for GET-only agent<br> \u2022 Complex JSON expected for simple calls<br> \
\u2022 Binary data handling for text-only agent|
|**coding-use-case-specific-failure-types**|coding-use-case-specific-failure-types-category-edge-case-oversights|\
Failure to handle non-standard inputs, boundary conditions, or exceptional scenarios in code generation or \
modification||
||coding-use-case-specific-failure-types-category-dependency-issues|Failures in handling code dependencies, \
imports, or external library requirements||"""

OUTPUT_FORMAT = """\
```json
{
  "errors": [
    {
      "location": "the span ID where failure occurred",
      "category": ["category name"],
      "evidence": ["brief explanation of the failure"],
      "confidence": ["low | medium | high"]
    }
  ]
}
```"""

FORMAT_REQUIREMENTS = """\
- Return valid JSON only, no additional text
- Use the exact field names: "location", "category", "evidence", "confidence"
- "category" must be an array of strings
- "evidence" must be an array of strings
- "confidence" must be an array where each element is one of: "low", "medium", "high"
- When multiple failure modes exist at the same location, arrays must maintain \
element-wise correspondence: category[i] corresponds to evidence[i] and confidence[i]
- If no failures are detected, return `{"errors": []}`"""

EXAMPLES = """\
### Single Failure Mode Example
If an agent repeatedly calls the same API without justification:
```json
{
  "errors": [
    {
      "location": "span_id_where_repetition_occurs",
      "category": ["repetitive-behavior-category-repetition-tool"],
      "evidence": ["Agent called check_flight_status API 3 times with identical parameters \
without user request or justification"],
      "confidence": ["high"]
    }
  ]
}
```

### Multiple Failure Modes at Same Location Example
If an agent fabricates tool output AND repeats the same incorrect action at the same span:
```json
{
  "errors": [
    {
      "location": "span_id_with_multiple_failures",
      "category": ["hallucination-category-hall-usage", "repetitive-behavior-category-repetition-tool"],
      "evidence": ["Agent fabricated a successful API response claiming flight was booked", \
"Agent repeated the same invalid booking attempt 3 times without addressing the underlying error"],
      "confidence": ["high", "high"]
    }
  ]
}
```

**Note:** When multiple failure modes occur at the same location, maintain element-wise \
correspondence across arrays: the first category corresponds to the first evidence and \
first confidence, and so on."""

QUALITY_CHECKS = """\
- All identified failures have clear evidence
- Categories accurately describe the failure type
- Confidence levels reflect certainty of the failure
- JSON is valid and follows the exact schema
- Location IDs match actual spans in the trajectory"""

TASK_INSTRUCTIONS = """\
Now analyze the session data provided between <session> and </session>. The session \
contains traces, each with spans representing atomic operations. Return your annotation \
in the exact JSON format specified above, without additional text or markdown formatting \
like ```json"""


def build_prompt(
    session_json: str,
    annotation_guidelines: str | None = None,
    category_descriptions: str | None = None,
) -> str:
    """Build the user message with session data for failure analysis.

    Whitespace is intentionally shaped to match Lens's Jinja template output
    byte-for-byte (with the same inputs). Jinja's ``{% if %}/{% else %}/{% endif %}``
    blocks leave an extra blank line between section headers and their bodies;
    reproducing that structure keeps the LLM's section-boundary emphasis
    identical to Lens so category attribution stays consistent.

    Args:
        session_json: Serialized session trace as JSON string.
        annotation_guidelines: Custom guidelines. Uses DEFAULT_ANNOTATION_GUIDELINES if None.
        category_descriptions: Custom taxonomy. Uses CATEGORIES if None.
    """
    guidelines = annotation_guidelines or DEFAULT_ANNOTATION_GUIDELINES
    categories = category_descriptions or CATEGORIES

    # Each section follows the pattern Jinja produces: header, blank line,
    # optional preamble, blank line, body, blank line, separator.
    return f"""---

## Annotation Guidelines


{guidelines}


---

## Failure Category Definitions


{categories}


---

## Output Format

Your output must be a valid JSON object following this exact structure:


{OUTPUT_FORMAT}


### Format Requirements:


{FORMAT_REQUIREMENTS}


---

## Examples


{EXAMPLES}


---

## Quality Checks

Ensure these are verified before submitting your annotation:


{QUALITY_CHECKS}


---

## Task Instructions


{TASK_INSTRUCTIONS}


---

<session>
{session_json}
</session>"""
