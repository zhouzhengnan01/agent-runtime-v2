# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Git Commit Rules

**NEVER mention "Claude", "Claude.ai", "Claude Code", or any Claude-related terms in git commit messages, code comments, or any files that will be committed to the repository.**

When creating git commits:
- Use generic terms like "Update", "Fix", "Add", "Refactor"
- Focus on what was changed, not who/what made the changes
- Avoid any AI/assistant references

## IMPORTANT: Git Push Strategy

**ALWAYS push to BOTH GitHub and Gitee when updating code.**

Remote repositories:
- **origin** (GitHub): `git@github.com:jetlinks-v2/jetlinks-agent-runtime.git`
- **gitee** (Gitee): `https://gitee.com/BetaStreetOmnis/jetlinks-agent-runtime.git`

When pushing changes:
```bash
# ALWAYS push to both remotes
git push origin langchain
git push gitee langchain
```

Or use a single command to push to both:
```bash
git push origin langchain && git push gitee langchain
```

**Why both remotes:**
- GitHub: Primary development repository, international access
- Gitee: China mirror, faster domestic access, backup
- Both must stay synchronized to ensure team access

## Project Overview

JetLinks Agent is an enterprise-grade AI Agent orchestration service module, split from the OtterFlowAI project. It provides commercial Agent capabilities while using internal APIs to call the open-source knowledge base module (jetlinks-knowledge) for RAG capabilities.

## Core Architecture

### Module Separation
- **Agent Module (this project)**: Agent management, tool execution, JAIP protocol, application orchestration
- **Knowledge Module (jetlinks-knowledge)**: RAG retrieval, document processing, vector storage, knowledge graph (accessed via internal API)

### Key Components
1. **JAIP Protocol Handler** (`app/core/jaip/handler.py`): Main entry point for WebSocket communication using JAIP v2.0 protocol (LangChain-based, formerly `langchain_handler.py`)
2. **Agent Templates** (LangChain-based):
   - **Video Inspection** (`video_inspection_template.py`): LangChain-based video analysis agent
   - **Tool Calling** (`tool_calling_template.py`): LangChain-based async tool execution with user confirmation
3. **LangChain Cognitive Agent** (`app/core/agents/cognitive_agent.py`): Async execution engine using LangChain (formerly `langchain_cognitive_agent.py`)
4. **LangChain Template Factory** (`app/core/agents/langchain_template_agent_factory.py`): Factory for creating LangChain-based agents
5. **Tool System** (`app/core/tools/`): Extensible tool registry for agent capabilities (still uses qwen-agent BaseTool interface)
6. **LLM Integration** (`app/core/llm/`): Multi-model support with DashScope and OpenAI
7. **Memory System** (`app/core/memory/unified_manager.py`): MySQL + Milvus hybrid storage for agent memory
8. **Storage System** (`app/core/storage/`): Session-based file organization and management
9. **Multimodal Processing** (`app/core/multimodal/`): Image/video analysis with vision models

## Development Commands

### Running the Server
```bash
# Start the server (recommended way)
./deploy/jetlinks-agent/start_server.sh

# Or directly with Python
python main.py

# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8005

# Server runs on port 8005 by default (configured in main.py)
```

### Database Operations
```bash
# Initialize database
alembic init alembic

# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Required environment variables
export DASHSCOPE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export DB_HOST="localhost"
export DB_PORT="3306"
export DB_NAME="jetlinks_agent"
export KNOWLEDGE_SERVICE_URL="http://localhost:8001"
```

## Architecture Patterns

### Agent Template Pattern (Updated 2025-10-24)
The system uses **LangChain-only architecture**:
- **LangChain Framework**: All agent templates are LangChain-based
  - Fully async execution with AgentExecutor
  - No event loop blocking
  - OpenAI function calling support
  - Converts qwen-agent tools to LangChain StructuredTool format
- **Factory Pattern**: `LangChainTemplateAgentFactory` creates all agents
- **Current Templates**:
  - `video_inspection`: Video analysis with multimodal capabilities
  - `tool_calling`: General purpose tool-calling agent with user confirmation

### JAIP Protocol Flow (Updated 2025-11-24)
1. WebSocket connection established at `/api/v1/jaip/session/{agent_id}`
2. Messages are processed through `JAIPHandler.process_message()` in `app/core/jaip/handler.py`
3. All agent interactions use LangChain-based execution
4. Multimodal content (images/videos) automatically processed when detected

### Tool Registration
Tools are auto-discovered and registered in `app/core/tools/__init__.py`:
- All tools inherit from `BaseTool` (qwen-agent compatible)
- Tools are registered in `TOOL_REGISTRY` dictionary
- Agents dynamically load tools based on configuration

## Key Implementation Details

### Multimodal Processing Pipeline (2025-01-21 Update)
1. **URL Detection & Download**: Automatically identifies URLs in user messages, downloads images/videos
2. **File Upload Support**: Handles direct file uploads via `/api/v1/upload` endpoints
3. **Unified Media Analysis**:
   - Images: Direct analysis with qwen-vl-max or other vision models
   - Videos: Frame extraction (using existing VideoProcessor) → Multi-frame analysis
4. **Intelligent Inspection Triggering**:
   - Detects inspection keywords (巡检, 检查, 安全, 隐患, 风险, etc.)
   - Automatically invokes `unified_inspector` when media content is present
   - Generates formatted inspection reports with safety analysis
5. **Context Enhancement**: Augments user messages with media analysis results before agent processing

### File Handling Architecture (2025-01-21 Update)
**Layered architecture with clear separation of concerns:**

1. **File Service Layer** (`app/services/file_service.py`)
   - Unified file upload, storage, and access management
   - Generates unique file IDs for tracking
   - Provides dual access pattern:
     - **HTTP URL**: `/static/uploads/{type}/{filename}` for frontend access
     - **Absolute Path**: Local filesystem path for AI model processing
   - Automatic file type detection and categorization
   - File registry for metadata tracking

2. **Content Analysis Service** (`app/services/content_analyzer.py`)
   - Analyzes media content from URLs or file paths
   - Supports multiple content types:
     - Images: Uses vision models (qwen-vl-max)
     - Videos: Frame extraction and analysis
     - Documents: Text extraction
   - Async processing with fallback mechanisms
   - Integrates with multiple analyzer backends

3. **Message Processor** (`app/core/processor/message_processor.py`)
   - Orchestrates the entire message processing pipeline
   - Workflow:
     1. Receives message with optional file_id
     2. Retrieves file info from File Service
     3. Extracts URLs from message text
     4. Analyzes all media content via Content Analyzer
     5. Builds enhanced context with analysis results
     6. Constructs augmented message for agent processing
     7. Passes enhanced message to agent
   - Supports both sync and streaming modes

4. **API Integration**
   - **Files API** (`/api/v1/files/*`): REST endpoints for file operations
   - **Upload API** (`/api/v1/upload/*`): Legacy upload endpoints
   - **Static Files**: Mounted at `/static/uploads/` for direct access

5. **JAIP Integration**
   - WebSocket handler checks for `file_id` in message context
   - If present, routes through MessageProcessor for file handling
   - Otherwise uses existing MultimodalContentProcessor
   - Seamless integration with existing agent infrastructure

### Video Processing Components
- **Primary**: `app/shared/video/video_processor.py` - Main video processing with OpenCV
- **Advanced**: `app/shared/video/advanced_video_processor.py` - Enhanced frame extraction
- **Frame Analysis**: `app/shared/video/frame_analyzer.py` - Individual frame analysis
- **Multimodal**: `app/core/multimodal/` - New unified content processing module
  - `content_processor.py`: Orchestrates URL/upload processing
  - `url_resolver.py`: Downloads and identifies media from URLs  
  - `media_analyzer.py`: Analyzes images/videos with VLM models
  - `unified_inspector.py`: Unified inspection for safety analysis

### Session Management
- Sessions stored in-memory in `JAIPHandler.sessions`
- Each session maintains agent_id, context, and message history
- Video contexts are preserved for follow-up questions

### Conversation and File Storage Architecture (2025-01-25 Update)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            用户交互层                                      │
│                    (WebSocket / HTTP API / CLI)                          │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
┌────────────────────────────────┴───────────────────────────────────────┐
│                         JAIP Handler                                    │
│                 (消息处理 & 会话管理 & URL检测)                           │
│                    ↓                                                    │
│            检测到URL时自动触发下载                                        │
└────────────┬─────────────────────────────────┬────────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────┐
│   Conversation Manager      │   │      File Manager                   │
│   (对话历史管理)             │   │   (文件上传管理 & URL下载)           │
├─────────────────────────────┤   ├─────────────────────────────────────┤
│ • 保存对话轮次              │   │ • 文件上传处理                      │
│ • 生成Markdown              │   │ • URL检测和下载                     │
│ • 更新索引                  │   │ • 文件类型识别                      │
│ • 关联下载文件              │   │ • 存储路径管理                      │
└────────────┬────────────────┘   └────────────┬────────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        统一存储层                                         │
│                 storage/sessions/{agent_id}/{session_id}/               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────┐    ┌────────────────────────────┐        │
│  │    conversations/        │    │    files/{timestamp}/      │        │
│  ├──────────────────────────┤    ├────────────────────────────┤        │
│  │ • daily/                 │    │ • original/ (原始文件)      │        │
│  │   - 20250125.md         │    │ • markdown/ (转换文件)      │        │
│  │   - 20250126.md         │    │ • metadata.json            │        │
│  │ • summary.md            │    └────────────────────────────┘        │
│  │ • index.json            │                                           │
│  └──────────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
          ┌─────────▼──────────┐   ┌─────────▼──────────┐
          │  Markdown 转换器    │   │   检索 & 分析器     │
          ├────────────────────┤   ├────────────────────┤
          │ • PDF → Markdown   │   │ • 全文搜索         │
          │ • Video → Markdown │   │ • 时间范围查询      │
          │ • Image → Markdown │   │ • 标签过滤         │
          │ • Context → MD     │   │ • 智能总结         │
          └────────────────────┘   └────────────────────┘
```

**Storage Structure**:
```
storage/sessions/
└── {agent_id}/
    └── {session_id}/
        ├── conversations/    # 对话历史
        │   ├── daily/       # 每日记录
        │   ├── summary.md   # 总结
        │   └── index.json   # 索引
        └── files/           # 文件存储
            └── {timestamp}/ # 20250125_143025
                ├── original/
                └── markdown/
```

### New File Organization System (2025-01-25)

#### Core Design Principles
1. **Automatic URL Detection**: When users include URLs in messages, the system automatically downloads and processes files
2. **Unified Storage**: All files stored in `agent_id/session_id/timestamp/` structure
3. **Markdown Conversion**: All documents, videos, images converted to Markdown format
4. **Same-name Policy**: Markdown files have same name as originals (e.g., `report.pdf` → `report.md`)
5. **Conversation History**: All dialogues saved as Markdown in daily files

#### New Components Structure

**Already Created (5 files)**:
- ✅ `app/core/storage/__init__.py`
- ✅ `app/core/storage/session_storage.py` - Abstract base class
- ✅ `app/core/storage/file_organizer.py` - File organization logic
- ✅ `app/core/storage/markdown_templates.py` - Markdown templates
- ✅ `app/services/session_file_manager.py` - Main file management service

**To Be Created (35 files)**:

1. **Utils (4 files)**:
   - `app/utils/markdown_formatter.py` - Markdown formatting utilities
   - `app/utils/file_type_detector.py` - File type detection and classification
   - `app/utils/text_extractor.py` - Text extraction from various formats
   - `app/utils/media_utils.py` - Video/image processing utilities

2. **Converters (10 files)**:
   - `app/core/converters/base_converter.py` - Abstract converter base
   - `app/core/converters/unified_markdown_converter.py` - Router to specific converters
   - `app/core/converters/pdf_to_markdown.py` - PDF conversion
   - `app/core/converters/office_to_markdown.py` - Word/Excel/PPT conversion
   - `app/core/converters/video_to_markdown.py` - Video analysis and conversion
   - `app/core/converters/image_to_markdown.py` - Image OCR and scene analysis
   - `app/core/converters/text_to_markdown.py` - Plain text conversion
   - `app/core/converters/html_to_markdown.py` - HTML to Markdown
   - `app/core/converters/session_context_converter.py` - Session context export
   - Update `app/core/converters/__init__.py` - Converter registration

3. **Services (6 files)**:
   - `app/services/conversation_history_manager.py` - Save/retrieve conversations
   - `app/services/conversation_retriever.py` - Search conversation history
   - `app/services/conversation_summarizer.py` - Auto-generate summaries
   - `app/services/session_file_retriever.py` - File search and retrieval
   - `app/services/storage_manager.py` - Storage cleanup and management
   - `app/services/markdown_service.py` - Markdown operations service

4. **API Endpoints (3 files)**:
   - `app/api/v1/conversations.py` - Conversation history API
   - `app/api/v1/session_files.py` - Session file management API
   - `app/api/v1/markdown.py` - Markdown content API

5. **Data Models (3 files)**:
   - `app/schemas/conversation.py` - Conversation schemas
   - `app/schemas/session_file.py` - File metadata schemas
   - `app/schemas/markdown.py` - Markdown content schemas

**Files to Modify (6 files)**:
- `app/config.py` - Add storage path and conversion settings
- `app/main.py` - Register new API routes
- `app/core/jaip/handler.py` - Integrate conversation saving and URL detection
- `app/services/file_service.py` - Use new storage structure
- `app/api/v1/files.py` - Return Markdown paths
- `requirements.txt` - Add pypdf2, python-docx, openpyxl, etc.

#### Configuration Additions

```python
# app/config.py additions
STORAGE_BASE_PATH = os.getenv("STORAGE_BASE_PATH", "storage/sessions")
MARKDOWN_CONVERSION_ENABLED = os.getenv("MARKDOWN_CONVERSION_ENABLED", "true")
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", "30"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
AUTO_DOWNLOAD_URLS = os.getenv("AUTO_DOWNLOAD_URLS", "true")
```

#### Required Dependencies

```txt
# New dependencies for requirements.txt
pypdf2>=3.0.0              # PDF processing
python-docx>=0.8.11        # Word documents
openpyxl>=3.0.10          # Excel processing
python-pptx>=0.6.21       # PowerPoint processing
pytesseract>=0.3.10       # OCR for images
beautifulsoup4>=4.11.0    # HTML parsing
markdown>=3.4.0           # Markdown processing
python-magic>=0.4.27      # File type detection
```

### Error Handling
- All async operations wrapped in try-catch blocks
- Detailed logging at each processing stage
- User-friendly error messages returned via JAIP protocol

## Critical Dependencies

### Required Python Packages (Updated 2025-10-24)
- **langchain**: LangChain framework for all agent templates
- **langchain-community**: LangChain community integrations
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-core**: LangChain core library
- **qwen-agent**: Still used for BaseTool interface only (tools inherit from it)
- **dashscope**: Alibaba Cloud AI model SDK (for qwen-vl-max vision model)
- **fastapi**: Web framework
- **sqlalchemy**: Database ORM
- **uvicorn**: ASGI server
- **opencv-python**: Video frame extraction (optional, fallback available)
- **pillow (PIL)**: Image processing
- **aiohttp**: Async HTTP client for URL downloads

### External Services
- Knowledge Service API (jetlinks-knowledge) at port 8001
- MySQL database for agent configurations
- DashScope API for Qwen models

## Common Development Tasks

### Adding a New Tool
1. Create tool class in `app/core/tools/`
2. Inherit from `BaseTool` 
3. Implement `call()` method
4. Register in `TOOL_REGISTRY`

### Creating a New Agent Template (Updated 2025-10-24)
1. Create class in `app/core/agents/template_agent/`
2. Inherit from `BaseLangChainTemplateAgent`
3. Implement required abstract methods:
   - `_initialize()` - Initialize agent-specific resources
   - `process_message()` - Handle user messages
   - `execute_task()` - Execute specific tasks (optional)
   - `get_capabilities()` - Return capability list
4. Register in `LangChainTemplateAgentFactory.TEMPLATE_REGISTRY`
5. Create database configuration with matching `type` field

**Example: LangChain-based Template**
```python
class ToolCallingAgent(BaseLangChainTemplateAgent):
    def __init__(self, config):
        super().__init__(config)
        self.cognitive_engine = LangChainCognitiveAgent(config)
        self.function_list = self.cognitive_engine.function_list

    async def process_message(self, message, context):
        messages = [
            {"role": "system", "content": self.get_system_prompt(context)},
            {"role": "user", "content": message}
        ]
        result = await self.cognitive_engine._run_async(messages)
        return result[-1].content
```

### Modifying JAIP Protocol (Updated 2025-10-24)
1. Add handler method in `LangChainJAIPHandler`
2. Update WebSocket endpoint in `app/api/v1/websocket.py` or `websocket_auto.py` if needed
3. Protocol data models were removed - use direct message handling

## Project Structure Guidelines

### Directory Organization
- **Tests**: All test files must be placed in `/tests/` directory
  - Test files should start with `test_` prefix
  - The `.gitignore` already excludes all `test*` files
  - Run tests with `pytest tests/` or individual files
- **Documentation**: All documentation should be in `/docs/` directory
  - API documentation (e.g., WEBSOCKET_PROTOCOL.md)
  - Architecture diagrams
  - Protocol specifications
- **Source Code**: Application code in `/app/` directory
- **Static Files**: Generated files in `/static/` directory
- **Configuration**: Config files in root or `/config/` directory

### File Naming Conventions
- Test files: `test_*.py`
- Documentation: Use `.md` extension
- Python modules: lowercase with underscores
- Classes: PascalCase
- Functions/variables: snake_case

## Testing Approach

1. All tests are located in `/tests/` directory
2. Use pytest for unit tests
3. Mock external services (Knowledge API, LLM calls)
4. Test JAIP message handling with WebSocket test client
5. Validate tool execution independently

## Project Structure Updates

### Major Cleanup and Renames (2025-11-24)
**File Renames (Simplified naming):**
- `langchain_handler.py` → `handler.py` (JAIP protocol handler)
- `langchain_cognitive_agent.py` → `cognitive_agent.py` (Cognitive execution engine)

**Current Architecture (LangChain-only):**
1. **BaseLangChainTemplateAgent** - Base class for all agents
2. **LangChainCognitiveAgent** (in `cognitive_agent.py`) - Async execution engine with tool calling
3. **LangChainTemplateAgentFactory** - Factory for agent creation
4. **JAIPHandler** (in `handler.py`) - WebSocket protocol handler
5. **Tool System** - Still uses qwen-agent's `BaseTool` interface for compatibility

### Directory Reorganization (2025-01-16)
- **Video Processing**: Moved from `app/core/jaip/` to `app/shared/video/`
- **JAIP Handler**: `handler.py` is the main handler (simplified from `langchain_handler.py`)
- **Storage**: Session-based file organization in `app/core/storage/`
- **Multimodal**: Unified content processing in `app/core/multimodal/`

## Important Notes (Updated 2025-11-24)

- **LangChain-Only Architecture**: All agents are LangChain-based
- **File Naming**: Core files simplified (handler.py, cognitive_agent.py) without "langchain_" prefix
- **Tool Interface Compatibility**: Tools still use qwen-agent's `BaseTool` interface for backward compatibility
- **Async Execution**: All agent templates use async execution to avoid event loop blocking
- **Tool Calling**: All templates support WebSocket-based tool calling with user confirmation
- **DASHSCOPE_API_KEY**: Must be set for vision model (qwen-vl-max) to work
- **Absolute Imports**: Project uses absolute imports from `app` package
- **Session Data**: In-memory sessions will be lost on restart
- **Video Processing**: Files processed locally, ensure sufficient disk space
- **Memory System**: Requires MySQL and optionally Milvus for vector search
- **Storage System**: Session-based file organization in `storage/sessions/`

## Known Issues and Solutions (Updated 2025-10-24)

### ✅ RESOLVED: WebSocket Tool Confirmation Timeout
**Previous Problem**: qwen-agent framework blocked event loop, causing tool confirmation timeouts.

**Solution Implemented**: Migrated all agents to LangChain framework with async execution:
- All templates now use `LangChainCognitiveAgent` for async tool calling
- Tool confirmation messages sent in <0.1 seconds
- No event loop blocking
- qwen-agent framework completely removed from agent layer

### ✅ RESOLVED: Agent Not Found Error
**Previous Problem**: Agent model missing @property accessors for config fields.

**Solution Implemented**: Agent model now includes property accessors:
- `system_prompt`, `model`, `temperature`, `max_tokens`, `tools`, `user_prompt`
- All configuration stored in `config` JSON field
- Properties automatically extract values from config
