from unittest.mock import AsyncMock, patch

import pytest
from strands_evals.generators.topic_planner import Topic, TopicPlan, TopicPlanner


@pytest.mark.asyncio
async def test_topic_planner_plan_topics_async():
    """Test topic planning generates correct number of topics"""
    planner = TopicPlanner()

    mock_agent = AsyncMock()
    mock_plan = TopicPlan(
        topics=[
            Topic(title="Topic 1", description="Desc 1", key_aspects=["aspect1", "aspect2"]),
            Topic(title="Topic 2", description="Desc 2", key_aspects=["aspect3"]),
        ]
    )
    mock_agent.structured_output_async.return_value = mock_plan

    with patch("strands_evals.generators.topic_planner.Agent", return_value=mock_agent):
        result = await planner.plan_topics_async("context", "task", num_topics=2, num_cases=10)

    assert len(result.topics) == 2
    assert all(isinstance(t, Topic) for t in result.topics)
