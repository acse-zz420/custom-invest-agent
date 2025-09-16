import os
import logging
from typing import Optional
from dotenv import load_dotenv


class Neo4jConnector:
    """Neo4j数据库连接管理类"""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, auto_load_env: bool = True):
        """
        初始化Neo4j连接器
        Args:
            uri: Neo4j数据库URI，例如 'bolt://localhost:7687'
            user: 数据库用户名
            password: 数据库密码
            auto_load_env: 是否自动加载环境变量（如果参数未提供）
        """
        self.uri: Optional[str] = uri
        self.user: Optional[str] = user
        self.password: Optional[str] = password
        self.driver = None
        self._setup_logging()

        if auto_load_env:
            self._load_environment()
            self._validate_connection_params()

    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=print,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        self.logger = logging.getLogger(__name__)

    def _load_environment(self):
        """加载环境变量（仅在参数未提供时）"""
        load_dotenv()

        # 只有在参数未提供时才从环境变量加载
        if self.uri is None:
            self.uri = os.environ.get('NEO4J_URI')
        if self.user is None:
            self.user = os.environ.get('NEO4J_USER')
        if self.password is None:
            self.password = os.environ.get('NEO4J_PASSWORD')
        self.logger.info(f"Neo4j连接配置: {self.uri}, 用户: {self.user}")

    def _validate_connection_params(self):
        """验证连接参数"""
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Neo4j连接参数不完整。请提供uri、user和password参数，"
                "或设置环境变量NEO4J_URI、NEO4J_USER、NEO4J_PASSWORD"
            )

        return True

    def get_connection_params(self):
        """获取连接参数"""

        return {
            'uri': self.uri,
            'user': self.user,
            'password': self.password
        }

    def update_connection_params(self, uri: Optional[str] = None,
                                 user: Optional[str] = None,
                                 password: Optional[str] = None):
        """更新连接参数"""
        if uri is not None:
            self.uri = uri
        if user is not None:
            self.user = user
        if password is not None:
            self.password = password
        self._validate_connection_params()
        self.logger.info(f"连接参数已更新: {self.uri}, 用户: {self.user}")

    async def clean_database(self, driver):
        """清理数据库中的所有数据"""
        self.logger.info("清理数据库中的现有数据...")
        # 删除所有节点和关系
        cypher_queries = [
            "MATCH (n) DETACH DELETE n",
            "MATCH ()-[r]-() DELETE r"
        ]

        for query in cypher_queries:
            try:
                await driver.execute_query(query)
                self.logger.info(f"执行查询: {query}")
            except Exception as e:
                self.logger.warning(f"查询失败 (可能正常): {query} - {e}")
            self.logger.info("数据库清理完成")

    def validate_connection(self):
        """验证连接参数"""
        return self._validate_connection_params()
