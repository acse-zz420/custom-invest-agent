import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union

from graphiti_core.nodes import EpisodeType


class KnowledgeGraphBuilder:
    """通用知识图谱构建类"""

    def __init__(self, graphiti_manager):
        self.graphiti_manager = graphiti_manager
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    # 情节数据构造
    # ------------------------------------------------------------------ #
    def create_episode_data(
        self,
        content: Union[str, Dict[str, Any]],
        episode_type: EpisodeType = EpisodeType.text,
        description: str = "通用信息",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建通用情节数据"""
        start_time = time.time()

        if name is None:
            name = f"情节_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        result = {
            "content": content,
            "type": episode_type,
            "description": description,
            "name": name,
        }

        print(f"create_episode_data 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    def create_text_episode(
        self,
        text_content: str,
        description: str = "文本信息",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建文本情节数据"""
        start_time = time.time()

        result = self.create_episode_data(
            content=text_content,
            episode_type=EpisodeType.text,
            description=description,
            name=name,
        )

        print(f"create_text_episode 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    def create_json_episode(
        self,
        json_data: Dict[str, Any],
        description: str = "结构化信息",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建 JSON 情节数据"""
        start_time = time.time()

        result = self.create_episode_data(
            content=json_data,
            episode_type=EpisodeType.json,
            description=description,
            name=name,
        )

        print(f"create_json_episode 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    @staticmethod
    def convert_json_to_text(json_data: Dict[str, Any]) -> str:
        """将 JSON 数据转换为自然语言文本"""
        start_time = time.time()

        content_parts = []
        for key, value in json_data.items():
            if value is None:
                continue

            if isinstance(value, list):
                value_str = "、".join(str(item) for item in value)
            else:
                value_str = str(value)

            content_parts.append(f"{key}是{value_str}")

        result = "，".join(content_parts) + "。"

        print(f"convert_json_to_text 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    # ------------------------------------------------------------------ #
    # 写入 Graphiti
    # ------------------------------------------------------------------ #
    async def add_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        """批量添加情节到知识图谱"""
        start_time = time.time()

        if not episodes:
            self.logger.warning("没有提供情节数据")
            return

        self.logger.info(f"开始添加 {len(episodes)} 个情节...")
        graphiti = self.graphiti_manager.get_graphiti()

        for i, episode in enumerate(episodes, start=1):
            ep_start = time.time()
            try:
                content = episode["content"]
                episode_type = episode["type"]
                description = episode.get("description", "通用信息")
                name = episode.get("name", f"情节_{i}")

                # 处理内容格式
                if episode_type == EpisodeType.json and isinstance(content, dict):
                    episode_body = self.convert_json_to_text(content)
                elif isinstance(content, str):
                    episode_body = content
                else:
                    episode_body = json.dumps(content, ensure_ascii=False)

                await graphiti.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=episode_type,
                    source_description=description,
                    reference_time=datetime.now(timezone.utc),
                )

                self.logger.info(f"已添加情节: {name} ({episode_type.value})")
                print(f"添加情节 {i} ({name}) 耗时: {time.time() - ep_start:.4f} 秒")

            except Exception as e:
                self.logger.error(f"添加情节 {i} 时出错: {e}")
                raise

        print(f"add_episodes 总执行耗时: {time.time() - start_time:.4f} 秒")
        self.logger.info("所有情节添加完成")

    async def add_single_episode(
        self,
        content: Union[str, Dict[str, Any]],
        episode_type: EpisodeType = EpisodeType.text,
        description: str = "通用信息",
        name: Optional[str] = None,
    ) -> None:
        """添加单个情节"""
        start_time = time.time()

        if name is None:
            name = f"情节_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        graphiti = self.graphiti_manager.get_graphiti()

        try:
            # 处理内容格式
            if episode_type == EpisodeType.json and isinstance(content, dict):
                episode_body = self.convert_json_to_text(content)
            elif isinstance(content, str):
                episode_body = content
            else:
                episode_body = json.dumps(content, ensure_ascii=False)

            await graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source=episode_type,
                source_description=description,
                reference_time=datetime.now(timezone.utc),
            )

            self.logger.info(f"已添加情节: {name}")
            print(f"add_single_episode 执行耗时: {time.time() - start_time:.4f} 秒")

        except Exception as e:
            self.logger.error(f"添加情节时出错: {e}")
            raise

    async def add_json_episode(
        self,
        json_data: Dict[str, Any],
        description: str = "结构化信息",
        name: Optional[str] = None,
    ) -> None:
        """添加 JSON 格式的情节"""
        start_time = time.time()

        await self.add_single_episode(
            content=json_data,
            episode_type=EpisodeType.json,
            description=description,
            name=name,
        )

        print(f"add_json_episode 执行耗时: {time.time() - start_time:.4f} 秒")

    # ------------------------------------------------------------------ #
    # 批量构造辅助
    # ------------------------------------------------------------------ #
    def create_episode_batch(
        self,
        data_list: List[Any],
        episode_type: EpisodeType = EpisodeType.text,
        description: str = "批量信息",
    ) -> List[Dict[str, Any]]:
        """批量创建情节数据"""
        start_time = time.time()
        episodes = []

        for i, data in enumerate(data_list, start=1):
            episode = self.create_episode_data(
                content=data,
                episode_type=episode_type,
                description=description,
                name=f"批量情节_{i}",
            )
            episodes.append(episode)

        print(f"create_episode_batch 执行耗时: {time.time() - start_time:.4f} 秒")
        return episodes

    def create_text_episode_batch(
        self,
        text_list: List[str],
        description: str = "批量文本信息",
    ) -> List[Dict[str, Any]]:
        """批量创建文本情节数据"""
        start_time = time.time()

        result = self.create_episode_batch(
            data_list=text_list,
            episode_type=EpisodeType.text,
            description=description,
        )

        print(f"create_text_episode_batch 执行耗时: {time.time() - start_time:.4f} 秒")
        return result

    def create_json_episode_batch(
        self,
        json_list: List[Dict[str, Any]],
        description: str = "批量结构化信息",
    ) -> List[Dict[str, Any]]:
        """批量创建 JSON 情节数据"""
        start_time = time.time()

        result = self.create_episode_batch(
            data_list=json_list,
            episode_type=EpisodeType.json,
            description=description,
        )

        print(f"create_json_episode_batch 执行耗时: {time.time() - start_time:.4f} 秒")
        return result