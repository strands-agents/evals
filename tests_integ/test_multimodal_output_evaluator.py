"""Integration tests for MultimodalOutputEvaluator.

These tests make actual API calls to test the full multimodal evaluation workflow.
They are separate from unit tests to avoid unnecessary API costs during regular testing.

Real-world images are fetched once per session from picsum.photos (deterministic /id/ endpoints).
"""

import tempfile
import urllib.request
from pathlib import Path

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import MultimodalOutputEvaluator
from strands_evals.types.multimodal import ImageData, MultimodalInput

# Deterministic picsum.photos URLs — same image every time for a given ID.
# id/274: Times Square at night (city, neon signs, buildings, crowds)
# id/152: Purple petunia flowers (close-up, nature, green leaves)
CITY_IMAGE_URL = "https://picsum.photos/id/274/300/200.jpg"
FLOWER_IMAGE_URL = "https://picsum.photos/id/152/300/200.jpg"


# =============================================================================
# Session-scoped fixtures — images are downloaded once for the entire test run
# =============================================================================


@pytest.fixture(scope="session")
def city_bytes():
    """Download Times Square photo once per session."""
    with urllib.request.urlopen(CITY_IMAGE_URL, timeout=30) as resp:  # noqa: S310
        return resp.read()


@pytest.fixture(scope="session")
def flower_bytes():
    """Download flower photo once per session."""
    with urllib.request.urlopen(FLOWER_IMAGE_URL, timeout=30) as resp:  # noqa: S310
        return resp.read()


@pytest.fixture
def city_image(city_bytes):
    return ImageData(source=city_bytes, format="jpeg")


@pytest.fixture
def flower_image(flower_bytes):
    return ImageData(source=flower_bytes, format="jpeg")


@pytest.fixture
def temp_image_file(city_bytes):
    """Write downloaded image to a temp file for file-path tests."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(city_bytes)
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Integration tests - MLLM-as-a-Judge (with media)
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_basic_integration(city_image):
    """Test basic MLLM-as-a-Judge evaluation with a real photograph."""

    def multimodal_task(case: Case) -> str:
        return (
            "The image shows a bustling city scene at night, "
            "likely Times Square, with bright neon signs and advertisements."
        )

    test_case = Case(
        name="city_description",
        input=MultimodalInput(
            media=city_image,
            instruction="Describe what you see in this image.",
        ),
        expected_output=(
            "A nighttime city scene with illuminated billboards and signs."
        ),
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Score based on accuracy of the image description. "
            "1.0 if the description correctly identifies the main subject "
            "and key visual elements, "
            "0.5 if partially correct, 0.0 if completely wrong."
        ),
        include_media=True,
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(multimodal_task)

    assert len(reports[0].scores) == 1
    assert isinstance(reports[0].test_passes[0], bool)
    assert 0.0 <= reports[0].scores[0] <= 1.0


@pytest.mark.asyncio
async def test_multimodal_evaluator_reference_based(flower_image):
    """Test reference-based evaluation with a real flower photo."""

    def multimodal_task(case: Case) -> str:
        return "Purple petunia flowers with green leaves in close-up."

    test_case = Case(
        name="flower_reference",
        input=MultimodalInput(
            media=flower_image,
            instruction="Describe the main subject in this photograph.",
        ),
        expected_output="Purple flowers with green foliage.",
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Evaluate if the response correctly describes the image content. "
            "Score 1.0 for accurate descriptions, 0.0 for inaccurate ones."
        ),
        include_media=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(multimodal_task)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] >= 0.5


@pytest.mark.asyncio
async def test_multimodal_evaluator_reference_free(city_image):
    """Test reference-free evaluation — rubric only, no expected output."""

    def multimodal_task(case: Case) -> str:
        return (
            "This photograph captures a vibrant city streetscape at night. "
            "Tall buildings line the street, covered in illuminated billboards "
            "and neon signs. Crowds of people are visible at street level."
        )

    test_case = Case(
        name="reference_free",
        input=MultimodalInput(
            media=city_image,
            instruction="Describe this image in detail.",
        ),
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Evaluate the quality and accuracy of the image description. "
            "Score based on: clarity (0.3), accuracy (0.4), "
            "completeness (0.3). A good description should mention "
            "the main subject, setting, and notable visual details."
        ),
        include_media=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(multimodal_task)

    assert len(reports[0].scores) == 1
    assert isinstance(reports[0].test_passes[0], bool)


# =============================================================================
# Integration tests - LLM-as-a-Judge (text-only mode)
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_llm_mode(city_image):
    """Text-only evaluation mode where media is excluded from the judge prompt."""

    def multimodal_task(case: Case) -> str:
        return (
            "A brightly lit urban scene at night with tall buildings "
            "and illuminated commercial signage."
        )

    test_case = Case(
        name="llm_mode_test",
        input=MultimodalInput(
            media=city_image,
            instruction="Describe the image.",
        ),
        expected_output="A city at night with neon signs and buildings.",
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Compare the output against the expected output. "
            "Score 1.0 if semantically similar, "
            "0.0 if completely different."
        ),
        include_media=False,
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(multimodal_task)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] >= 0.5


# =============================================================================
# Integration tests - Multiple images
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_multiple_images(city_image, flower_image):
    """Test evaluation with two distinct real-world images."""

    def multi_image_task(case: Case) -> str:
        return (
            "The first image shows a city street at night with neon signs, "
            "and the second image shows purple flowers in close-up. "
            "They depict completely different subjects."
        )

    test_case = Case(
        name="multi_image",
        input=MultimodalInput(
            media=[city_image, flower_image],
            instruction="Compare these two images. What do they show?",
        ),
        expected_output=(
            "The first image is a city nightscape and the second "
            "is a close-up of flowers. They are unrelated subjects."
        ),
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Evaluate if the response correctly identifies and compares "
            "both images. Score 1.0 if both subjects are correctly "
            "identified, 0.0 if incorrect."
        ),
        include_media=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(multi_image_task)

    assert len(reports[0].scores) == 1


# =============================================================================
# Integration tests - File path input
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_with_file_path(temp_image_file):
    """Test evaluation with image loaded from a local file path."""
    image = ImageData(source=temp_image_file)

    def file_task(case: Case) -> str:
        return "A city street at night with neon signs and buildings."

    test_case = Case(
        name="file_path_test",
        input=MultimodalInput(
            media=image,
            instruction="Describe what this image shows.",
        ),
        expected_output="A nighttime city scene with illuminated signage.",
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Score 1.0 if the main subject of the image is correctly "
            "identified, 0.0 otherwise."
        ),
        include_media=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(file_task)

    assert len(reports[0].scores) == 1


# =============================================================================
# Integration tests - Sync evaluation
# =============================================================================


def test_multimodal_evaluator_sync(flower_image):
    """Test synchronous multimodal evaluation with a real photo."""

    def multimodal_task(case: Case) -> str:
        return "Purple flowers with green leaves."

    test_case = Case(
        name="sync_test",
        input=MultimodalInput(
            media=flower_image,
            instruction="Describe this image.",
        ),
        expected_output="Purple petunia flowers.",
    )

    evaluator = MultimodalOutputEvaluator(
        rubric="Score 1.0 if outputs match semantically, 0.0 otherwise.",
        include_media=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = experiment.run_evaluations(multimodal_task)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] >= 0.5


# =============================================================================
# Integration tests - Context handling
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_with_context(flower_image):
    """Test that evaluation context is passed through correctly."""

    def contextual_task(case: Case) -> str:
        return (
            "From a botanical perspective, the image shows petunia flowers "
            "(genus Petunia) with characteristic funnel-shaped corollas. "
            "The purple pigmentation suggests anthocyanin production."
        )

    test_case = Case(
        name="context_test",
        input=MultimodalInput(
            media=flower_image,
            instruction="Analyze this image.",
            context="You are a botanist identifying plant species.",
        ),
    )

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Evaluate if the response appropriately considers the given "
            "context. Score based on relevance to the specified domain "
            "and accuracy of observations."
        ),
        include_media=True,
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(contextual_task)

    assert len(reports[0].scores) == 1


# =============================================================================
# Integration tests - Multiple cases (batch)
# =============================================================================


@pytest.mark.asyncio
async def test_multimodal_evaluator_batch_processing(city_image, flower_image):
    """Test batch evaluation with multiple distinct test cases."""

    def batch_task(case: Case) -> str:
        if "city" in case.name:
            return "A city street at night with bright signs."
        elif "flower" in case.name:
            return "Purple flowers in close-up."
        return "Unknown image."

    test_cases = [
        Case(
            name="city_case",
            input=MultimodalInput(
                media=city_image,
                instruction="What is shown in this image?",
            ),
            expected_output="A nighttime city scene.",
        ),
        Case(
            name="flower_case",
            input=MultimodalInput(
                media=flower_image,
                instruction="What is shown in this image?",
            ),
            expected_output="Purple flowers.",
        ),
    ]

    evaluator = MultimodalOutputEvaluator(
        rubric=(
            "Score 1.0 if the output correctly identifies the subject, "
            "0.0 otherwise."
        ),
        include_media=True,
    )

    experiment = Experiment(cases=test_cases, evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(batch_task)

    assert len(reports[0].scores) == 2
    assert len(reports[0].test_passes) == 2
