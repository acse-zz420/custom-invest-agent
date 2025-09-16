import logging
from typing import Optional

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from config import config


class OllamaGraphitiManager:
    """Ollama LLM 连接和 Graphiti 初始化管理类"""

    def __init__(self, neo4j_connector):
        self.neo4j_connector = neo4j_connector
        self.graphiti: Optional[Graphiti] = None
        self.llm_client: Optional[OpenAIClient] = None
        self.embedder: Optional[OpenAIEmbedder] = None
        self.cross_encoder: Optional[OpenAIRerankerClient] = None
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # 配置部分
    # ------------------------------------------------------------------ #
    def setup_ollama_config(self) -> LLMConfig:
        """配置 Ollama LLM 客户端"""
        self.logger.info("配置 Ollama LLM 客户端...")

        # 从配置文件获取 Ollama 配置
        ollama_cfg = config.get_ollama_config()

        # 构造 LLMConfig
        llm_config = LLMConfig(
            api_key=ollama_cfg["api_key"],         # Ollama 实际不校验 API Key
            model=ollama_cfg["model"],
            small_model=ollama_cfg["small_model"],
            base_url=f"{ollama_cfg['base_url']}/v1",
        )

        # 初始化客户端
        self.llm_client = OpenAIClient(config=llm_config)
        self.logger.info("Ollama LLM 客户端配置完成")
        return llm_config

    def setup_embedder(self, llm_config: LLMConfig) -> OpenAIEmbedder:
        """配置 Embedder"""
        self.logger.info("配置 Embedder...")

        ollama_cfg = config.get_ollama_config()
        self.embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=ollama_cfg["api_key"],
                embedding_model=ollama_cfg["embedding_model"],
                embedding_dim=ollama_cfg["embedding_dim"],  # 例如 bge-large-zh-v1.5 -> 1024
                base_url=f"{ollama_cfg['base_url']}/v1",
            )
        )
        self.logger.info("Embedder 配置完成")
        return self.embedder

    def setup_cross_encoder(
        self, llm_client: OpenAIClient, llm_config: LLMConfig
    ) -> OpenAIRerankerClient:
        """配置 Cross Encoder"""
        self.logger.info("配置 Cross Encoder...")
        self.cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)
        self.logger.info("Cross Encoder 配置完成")
        return self.cross_encoder

    # ------------------------------------------------------------------ #
    # Graphiti 初始化 / 关闭
    # ------------------------------------------------------------------ #
    async def initialize_graphiti(self) -> Graphiti:
        """初始化 Graphiti 实例"""
        self.logger.info("初始化 Graphiti...")

        # 1. 连接 Neo4j
        conn_params = self.neo4j_connector.get_connection_params()

        # 2. 配置 LLM / Embedder / Cross-Encoder
        llm_config = self.setup_ollama_config()
        embedder = self.setup_embedder(llm_config)
        cross_encoder = self.setup_cross_encoder(self.llm_client, llm_config)

        # 3. 实例化 Graphiti
        self.graphiti = Graphiti(
            conn_params["uri"],
            conn_params["user"],
            conn_params["password"],
            llm_client=self.llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )
        self.logger.info("Graphiti 初始化完成")
        return self.graphiti

    async def close_connection(self) -> None:
        """关闭 Graphiti 连接"""
        if self.graphiti:
            await self.graphiti.close()
            self.logger.info("Graphiti 连接已关闭")

    # ------------------------------------------------------------------ #
    # 数据库操作
    # ------------------------------------------------------------------ #
    async def setup_database(self) -> None:
        """清理 Neo4j 并创建索引 / 约束"""
        if not self.graphiti:
            raise ValueError("Graphiti 未初始化")

        self.logger.info("设置数据库...")
        await self.neo4j_connector.clean_database(self.graphiti.driver)
        await self.graphiti.build_indices_and_constraints()
        self.logger.info("数据库设置完成")

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def get_graphiti(self) -> Graphiti:
        """获取 Graphiti 实例"""
        if not self.graphiti:
            raise ValueError("Graphiti 未初始化")
        return self.graphiti