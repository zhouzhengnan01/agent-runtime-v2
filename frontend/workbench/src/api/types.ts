export type AgentSummary = {
  name: string;
  display_name?: string;
  description?: string;
};

export type AppTemplate = {
  name: string;
  title?: string;
  description?: string;
  category?: string;
  icon?: string;
  agent_name?: string;
  workflow?: string;
  selected_skills?: string[];
  selected_mcp_tools?: string[];
  prompt_examples?: string[];
  tags?: string[];
  model_tags?: string[];
  models?: Array<Record<string, unknown>>;
  runtime_options?: Record<string, unknown>;
};

export type PendingAttachmentRequirement = {
  templateName: string;
  label: string;
  accept: string;
  reason: string;
};

export type ChatMessage = {
  role: "user" | "assistant" | "system";
  content: string;
};

export type AttachmentRef = {
  name?: string;
  original_name?: string;
  path?: string;
  url?: string;
  mime_type?: string | null;
  size?: number;
};

export type RuntimeOptions = {
  thread_id?: string;
  threadId?: string;
  selected_skills?: string[];
  selectedSkills?: string[];
  selected_mcp_tools?: string[];
  selectedMcpTools?: string[];
  app_template_name?: string;
  appTemplateName?: string;
  model_type?: string;
  modelType?: string;
  workflow?: string;
  mode?: "plan" | "edit" | "autonomous" | "safe" | "yolo";
  config_options?: Record<string, unknown>;
  configOptions?: Record<string, unknown>;
};

export type RunRequest = {
  messages: ChatMessage[];
  attachments?: AttachmentRef[];
  runtime_options?: RuntimeOptions;
};

export type Artifact = {
  name: string;
  path: string;
  kind: string;
  mime_type?: string;
  size?: number;
  preview_url: string;
  download_url: string;
};

export type VerificationResult = {
  passed?: boolean;
  retry_count?: number;
  checks?: Array<{
    name: string;
    passed?: boolean;
    detail?: string;
  }>;
};

export type RuntimeEvent = {
  type: string;
  data?: Record<string, unknown>;
};

export type AgentRunResult = {
  reply?: string;
  artifacts?: Artifact[];
  verification?: VerificationResult;
  spec?: unknown;
  metadata?: {
    requires_input?: boolean;
    required_inputs?: Array<Record<string, unknown>>;
  };
};

export type RequiredInput = {
  type?: string;
  accept?: string;
  reason?: string;
  description?: string;
  name?: string;
};

export type UploadItem = {
  id: string;
  name: string;
  displayName?: string;
  relativePath?: string;
  size: number;
  mime_type?: string | null;
  uploading?: boolean;
  progress?: number;
  error?: string;
  path?: string;
};

export type UploadBatchSummary = {
  total: number;
  completed: number;
  failed: number;
  uploading: number;
};

export type TimelineItem = {
  id: string;
  name: string;
  detail: string;
};

export type WorkbenchMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  progress?: boolean;
};
