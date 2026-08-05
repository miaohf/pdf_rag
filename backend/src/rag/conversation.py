"""
多轮对话管理系统

为Agentic RAG提供对话状态管理和上下文维护能力。
"""

import uuid
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy import func
from src.utils.logger import get_logger
from src.utils.config import Config
from src.core.database import Database

logger = get_logger(__name__)


class MessageType(Enum):
    """消息类型"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationState(Enum):
    """对话状态"""
    ACTIVE = "active"
    WAITING = "waiting"
    CLARIFYING = "clarifying"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class Message:
    """对话消息"""
    id: str
    type: MessageType
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    tool_calls: List[Dict[str, Any]] = None
    tool_results: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
            "tool_calls": self.tool_calls or [],
            "tool_results": self.tool_results or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建"""
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", [])
        )


@dataclass
class ConversationContext:
    """对话上下文"""
    topic: Optional[str] = None
    entities: List[str] = None
    intent: Optional[str] = None
    query_types: List[str] = None
    confidence: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.query_types is None:
            self.query_types = []
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()


class Conversation:
    """对话会话"""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.messages: List[Message] = []
        self.context = ConversationContext()
        self.state = ConversationState.ACTIVE
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, message: Message):
        """添加消息"""
        self.messages.append(message)
        self.last_activity = datetime.utcnow()
        self._update_context(message)
    
    def _update_context(self, message: Message):
        """更新对话上下文"""
        if message.type == MessageType.USER:
            # 更新主题和实体
            content = message.content.lower()
            
            # 简单的实体提取
            entities = self._extract_entities(content)
            for entity in entities:
                if entity not in self.context.entities:
                    self.context.entities.append(entity)
            
            # 更新查询类型
            if message.metadata and "query_type" in message.metadata:
                query_type = message.metadata["query_type"]
                if query_type not in self.context.query_types:
                    self.context.query_types.append(query_type)
            
            self.context.last_updated = datetime.utcnow()
    
    def _extract_entities(self, content: str) -> List[str]:
        """简单的实体提取"""
        # 这里可以集成更复杂的NER模型
        entities = []
        
        # 法律相关实体
        legal_terms = ["法律", "法规", "条例", "规定", "办法", "细则", "标准"]
        for term in legal_terms:
            if term in content:
                entities.append(term)
        
        # 技术相关实体
        tech_terms = ["系统", "技术", "算法", "设备", "平台", "软件", "硬件"]
        for term in tech_terms:
            if term in content:
                entities.append(term)
        
        return entities
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """获取最近的消息"""
        return self.messages[-limit:] if self.messages else []
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self.messages:
            return "新对话"
        
        summary_parts = []
        
        if self.context.topic:
            summary_parts.append(f"主题: {self.context.topic}")
        
        if self.context.entities:
            summary_parts.append(f"涉及: {', '.join(self.context.entities[:5])}")
        
        if self.context.query_types:
            summary_parts.append(f"查询类型: {', '.join(self.context.query_types)}")
        
        return " | ".join(summary_parts) if summary_parts else "一般对话"
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """检查对话是否过期"""
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.utcnow() - self.last_activity > timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "context": asdict(self.context),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, config: Config, database: Optional[Database] = None):
        self.config = config
        self.db = database
        self.active_conversations: Dict[str, Conversation] = {}
        self.session_timeout = 60  # 分钟
        
        logger.info("对话管理器初始化完成")
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新的对话会话"""
        session_id = str(uuid.uuid4())
        conversation = Conversation(session_id, user_id)
        self.active_conversations[session_id] = conversation
        
        logger.info(f"创建新对话会话: {session_id}")
        return session_id
    
    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """获取对话会话"""
        conversation = self.active_conversations.get(session_id)
        
        if conversation and conversation.is_expired(self.session_timeout):
            conversation.state = ConversationState.EXPIRED
            self._save_conversation(conversation)
            del self.active_conversations[session_id]
            return None
        
        return conversation
    
    def add_user_message(self, session_id: str, content: str, metadata: Dict[str, Any] = None) -> Message:
        """添加用户消息"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            # 自动创建新会话
            session_id = self.create_session()
            conversation = self.get_conversation(session_id)
        
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.USER,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        conversation.add_message(message)
        return message
    
    def add_assistant_message(
        self, 
        session_id: str, 
        content: str, 
        metadata: Dict[str, Any] = None,
        tool_calls: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None
    ) -> Message:
        """添加助手消息"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            logger.warning(f"会话不存在: {session_id}")
            return None
        
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.ASSISTANT,
            content=content,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            tool_calls=tool_calls or [],
            tool_results=tool_results or []
        )
        
        conversation.add_message(message)
        return message
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Message]:
        """获取对话历史"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            return []
        
        return conversation.get_recent_messages(limit)
    
    def build_conversation_context(self, session_id: str) -> str:
        """构建对话上下文字符串"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            return ""
        
        context_parts = []
        
        # 添加对话摘要
        summary = conversation.get_context_summary()
        if summary:
            context_parts.append(f"对话背景: {summary}")
        
        # 添加最近的对话历史
        recent_messages = conversation.get_recent_messages(6)  # 最近3轮对话
        if recent_messages:
            context_parts.append("\n最近对话:")
            for msg in recent_messages[-6:]:  # 限制历史长度
                role = "用户" if msg.type == MessageType.USER else "助手"
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def should_clarify(self, session_id: str, user_input: str) -> Tuple[bool, Optional[str]]:
        """判断是否需要澄清问题"""
        conversation = self.get_conversation(session_id)
        if not conversation:
            return False, None
        
        # 检查是否有模糊表达
        ambiguous_patterns = [
            "那个", "这个", "它", "刚才的", "上面的", "前面说的"
        ]
        
        for pattern in ambiguous_patterns:
            if pattern in user_input:
                # 尝试从上下文中找到指代
                recent_entities = conversation.context.entities[-3:] if conversation.context.entities else []
                if recent_entities:
                    suggestion = f"您是指 {', '.join(recent_entities)} 中的哪一个？"
                    return True, suggestion
                else:
                    return True, "请您说得更具体一些，我需要更多信息来帮助您。"
        
        return False, None
    
    def _save_conversation(self, conversation: Conversation):
        """保存对话到数据库（使用ORM）"""
        if not self.db:
            return
        
        try:
            from src.core.models import Conversation as ConversationModel
            from sqlalchemy.dialects.postgresql import insert
            
            with self.db.get_session() as session:
                # 使用ORM和upsert操作
                conversation_data = ConversationModel(
                    session_id=conversation.session_id,
                    user_id=conversation.user_id,
                    conv_metadata={
                        'conversation_data': conversation.to_dict(),
                        'state': conversation.state.value,
                        'created_at': conversation.created_at.isoformat() if conversation.created_at else None,
                        'last_activity': conversation.last_activity.isoformat() if conversation.last_activity else None
                    }
                )
                
                # PostgreSQL upsert (ON CONFLICT DO UPDATE)
                stmt = insert(ConversationModel).values(
                    session_id=conversation.session_id,
                    user_id=conversation.user_id,
                    conv_metadata=conversation_data.conv_metadata
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['session_id'],
                    set_=dict(
                        conv_metadata=stmt.excluded.conv_metadata,
                        updated_time=func.now()
                    )
                )
                session.execute(stmt)
                
                logger.debug(f"保存对话: {conversation.session_id}")
                
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
    
    def cleanup_expired_conversations(self):
        """清理过期对话"""
        expired_sessions = []
        
        for session_id, conversation in self.active_conversations.items():
            if conversation.is_expired(self.session_timeout):
                conversation.state = ConversationState.EXPIRED
                self._save_conversation(conversation)
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_conversations[session_id]
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期对话")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """获取对话统计信息"""
        return {
            "active_conversations": len(self.active_conversations),
            "total_messages": sum(len(conv.messages) for conv in self.active_conversations.values()),
            "average_messages_per_conversation": (
                sum(len(conv.messages) for conv in self.active_conversations.values()) / 
                len(self.active_conversations) if self.active_conversations else 0
            )
        }


class MultiTurnRagProcessor:
    """多轮RAG处理器 - 负责会话上下文，实际检索由 RAGService 编排"""
    
    def __init__(self, conversation_manager: ConversationManager):
        self.conversation_manager = conversation_manager
        logger.info("多轮RAG处理器初始化完成")
    
    def process_user_input(
        self, 
        user_input: str, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """处理用户输入（多轮对话版本）"""
        
        # 创建或获取会话
        if not session_id:
            session_id = self.conversation_manager.create_session(user_id)
        
        # 检查是否需要澄清
        needs_clarification, clarification = self.conversation_manager.should_clarify(session_id, user_input)
        if needs_clarification:
            # 添加用户消息
            self.conversation_manager.add_user_message(session_id, user_input)
            
            # 添加澄清请求
            self.conversation_manager.add_assistant_message(
                session_id, 
                clarification,
                metadata={"type": "clarification", "needs_user_response": True}
            )
            
            return {
                "session_id": session_id,
                "needs_clarification": True,
                "clarification": clarification,
                "response": clarification
            }
        
        # 添加用户消息到对话历史
        user_message = self.conversation_manager.add_user_message(session_id, user_input)
        
        # 构建对话上下文
        conversation_context = self.conversation_manager.build_conversation_context(session_id)
        
        return {
            "session_id": session_id,
            "needs_clarification": False,
            "conversation_context": conversation_context,
            "user_message": user_message,
            "ready_for_rag": True
        }
    
    def enhance_query_with_context(self, query: str, conversation_context: str) -> str:
        """使用对话上下文增强查询"""
        if not conversation_context:
            return query
        
        enhanced_query = f"""
{conversation_context}

当前问题: {query}

请结合上述对话背景来理解和回答当前问题。如果当前问题中有指代词（如"它"、"这个"、"那个"），请结合对话历史来理解其具体指代内容。
"""
        return enhanced_query
    
    def finalize_response(
        self, 
        session_id: str, 
        response: str, 
        metadata: Dict[str, Any] = None,
        tool_calls: List[Dict[str, Any]] = None,
        tool_results: List[Dict[str, Any]] = None
    ) -> Message:
        """完成响应并保存到对话历史"""
        
        return self.conversation_manager.add_assistant_message(
            session_id=session_id,
            content=response,
            metadata=metadata,
            tool_calls=tool_calls,
            tool_results=tool_results
        ) 