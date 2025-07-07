import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI

from trialmatcher import config
from trialmatcher.utils.retry_with_backoff import retry_with_exponential_backoff
from trialmatcher.utils.schemas import TrialMatcherConfig


class AzureClient:
    def __init__(
        self,
        run_config: TrialMatcherConfig,
        azure_endpoint: str = None,
        azure_api_key: str = None,
    ):
        """
        Class for managing OpenAI API clients.
        """
        self.run_config = run_config
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self._azure_client = None
        self._langchain_azure_openai_embeddings = None
        self._langchain_azure_openai_chat = None

    @property
    def azure_client(self) -> AzureOpenAI:
        """
        Returns the Azure OpenAI client.
        """
        if self._azure_client is None:
            endpoint = self.azure_endpoint or config.AZURE_OPENAI_API_ENDPOINT
            api_key = self.azure_api_key or config.AZURE_OPENAI_API_KEY or os.environ["AZURE_OPENAI_API_KEY"]

            if not endpoint:
                raise ValueError("Missing Azure endpoint")
            if not api_key:
                raise ValueError("Missing Azure API key")

            # https://github.com/openai/openai-python?tab=readme-ov-file#microsoft-azure-openai
            self._azure_client = AzureOpenAI(
                # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
                api_version=self.run_config.openai_api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
        return self._azure_client

    def chat_completions_parse(self, *args, **kwargs):
        """
        Wrapper around Azure OpenAI chat completions parse method without retry and backoff.
        """

        # Defining a local function and decorating it on the fly lets us use instance-specific values to parameterize the decorator.
        @retry_with_exponential_backoff(
            max_retries=self.run_config.max_retries, base_wait=self.run_config.base_wait
        )
        def wrapped_chat_completions_parse(*args, **kwargs):
            return self.azure_client.beta.chat.completions.parse(*args, **kwargs)

        return wrapped_chat_completions_parse(*args, **kwargs)

    @property
    def langchain_azure_openai_embeddings(self) -> AzureOpenAIEmbeddings:
        """
        Returns the Langchain AzureOpenAIEmbeddings object
        """
        if self._langchain_azure_openai_embeddings is None:
            self._langchain_azure_openai_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=self.run_config.embedding_model,
                openai_api_version=self.run_config.openai_api_version,
                api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_API_ENDPOINT,
            )
        return self._langchain_azure_openai_embeddings

    @property
    def langchain_azure_openai_chat(self) -> AzureChatOpenAI:
        """
        Returns the Langchain AzureChatOpenAI object
        """
        if self._langchain_azure_openai_chat is None:
            self._langchain_azure_openai_chat = AzureChatOpenAI(
                openai_api_version=self.run_config.openai_api_version,
                azure_deployment=self.run_config.llm_model,
                api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_API_ENDPOINT,
                temperature=0,
            )
        return self._langchain_azure_openai_chat
