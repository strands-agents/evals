from typing_extensions import TypeVar, Generic
from dataclasses import dataclass
import inspect

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")

from ..types.evaluation import EvaluationData, EvaluationOutput

@dataclass
class Evaluator(Generic[InputT, OutputT]):
    """
    Base class for evaluators.

    Evaluators can assess the performance of a task on all test cases.
    Subclasses must implement the `evaluate` method.
    """
    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> EvaluationOutput:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError("This method should be implemented in subclasses, especially if you want to run evaluations asynchronously.")
    
    @classmethod
    def get_type_name(cls) -> str:
        """
        Get the name of the evaluator type.

        Returns:
            str: The name of the evaluator type.
        """
        return cls.__name__

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.
        
        Returns:
            dict: A dictionary containing the evaluator's information. Omit private attributes
            (attributes starting with '_') and attributes with default values.
        """
        _dict = {"evaluator_type": self.get_type_name()}

        # Get default values from __init__ signature
        sig = inspect.signature(self.__init__)
        defaults = {k: v.default for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and (k not in defaults or v != defaults[k]):
                _dict[k] = v
        return _dict

    
if __name__ == "__main__":
    pass
