import math
from typing import List
from pydantic import BaseModel, Field
from strands import Agent


class Topic(BaseModel):
    """Represents a single topic for test case generation."""
    title: str = Field(..., description="Brief descriptive title for the topic")
    description: str = Field(..., description="Short description explaining the topic")
    key_aspects: List[str] = Field(
        ..., 
        description="2-5 key aspects that test cases should explore for this topic"
    )


class TopicPlan(BaseModel):
    """Represents a complete topic plan with multiple topics."""
    topics: List[Topic] = Field(
        ..., 
        description="List of diverse topics for comprehensive test coverage"
    )


class TopicPlanner:
    """Plans diverse topics for test case generation based on agent context."""
    
    DEFAULT_PLANNING_PROMPT = """You are a test scenario planner for AI agents. 
Your role is to analyze agent configurations and generate strategic topic plans 
that comprehensively evaluate agent capabilities.

Your topics should:
- Cover different aspects of the agent's capabilities
- Test edge cases and common scenarios
- Vary in complexity and scope
- Ensure comprehensive coverage of available tools and features
- Be diverse and non-overlapping"""
    
    def __init__(self, model: str = None, planning_prompt: str = None):
        self.model = model
        self.planning_prompt = planning_prompt or self.DEFAULT_PLANNING_PROMPT
    
    async def plan_topics_async(
        self,
        context: str,
        task_description: str,
        num_topics: int,
        num_cases: int
    ) -> TopicPlan:
        """Generate a strategic plan of diverse topics for test case generation."""
        cases_per_topic = math.ceil(num_cases / num_topics)
        
        planning_agent = Agent(
            model=self.model,
            system_prompt=self.planning_prompt,
            callback_handler=None
        )
        
        prompt = f"""Generate {num_topics} diverse topics for creating {num_cases} test cases.

Agent Context:
{context}

Task Description:
{task_description}

Requirements:
- Create exactly {num_topics} distinct topics
- Each topic will generate approximately {cases_per_topic} test cases
- Include 2-5 key aspects per topic that test cases should explore
- Ensure topics span different complexity levels and use cases
- Make topics diverse and non-overlapping"""
        
        topic_plan = await planning_agent.structured_output_async(TopicPlan, prompt)
        
        if len(topic_plan.topics) > num_topics:
            topic_plan.topics = topic_plan.topics[:num_topics]
        
        return topic_plan
