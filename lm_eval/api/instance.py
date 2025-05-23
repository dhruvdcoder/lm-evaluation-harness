from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


OutputType = Literal[
    "loglikelihood",
    "loglikelihood_rolling",
    "generate_until",
    "multiple_choice",
]


@dataclass
class Instance:
    """
    Attributes:
        request_type: The type of request to make to the model.
            Like generate_until, loglikelihood, etc.
        doc (dict): The document to make the request to
        arguments (tuple): The arguments to send to the model, typically the first entry
            is the prompt, and the rest are sampling related arguments for the model.
        idx (int): The index of the document
        metadata (Tuple[Optional[str], Optional[int], Optional[int]]): The metadata of the document
            task_name (str): The name of the task
            doc_id (int): The id of the document
            repeats (int): The number of times to repeat the document
        resps (list): The responses from the model
        filtered_resps (dict): The filtered responses from the model
    """

    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: Tuple[Optional[str], Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments
            if isinstance(self.arguments, tuple)
            else (self.arguments,)
        )
