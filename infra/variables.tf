variable "aws_region" {
  description = "AWS region for everything except CloudFront (global)."
  type        = string
  default     = "eu-central-1"
}

variable "project" {
  description = "Name prefix for all resources."
  type        = string
  default     = "shopware-docs-rag"
}

variable "github_repo" {
  description = "GitHub repository (owner/name) allowed to deploy through OIDC."
  type        = string
  default     = "Besirii/shopware-docs-rag"
}

variable "deploy_branch" {
  description = "Only workflows running on this branch may assume the deploy role."
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = "Set to false if the AWS account already has the GitHub OIDC provider (only one per account is allowed)."
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Email address for the monthly budget alerts."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly AWS budget for the whole account; alerts at 50% (actual) and 100% (forecast)."
  type        = number
  default     = 30
}

variable "qdrant_url" {
  description = "Qdrant Cloud cluster URL, e.g. https://xxxx.eu-central-1-0.aws.cloud.qdrant.io:6333"
  type        = string
}

variable "collection_name" {
  description = "Qdrant collection with the heading-chunked index."
  type        = string
  default     = "shopware_docs"
}

variable "llm_model" {
  description = "Anthropic model used by /ask and /agent."
  type        = string
  default     = "claude-haiku-5-5"
}

variable "cpu" {
  description = "Fargate CPU units (512 = 0.5 vCPU). Embedding a query and the reranker run on this CPU."
  type        = number
  default     = 512
}

variable "memory" {
  description = "Fargate memory in MiB. The three ONNX models plus Python need about 1.5 GB."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Number of API tasks. 0 pauses the API (the load balancer still costs money)."
  type        = number
  default     = 1
}

variable "image_tag" {
  description = "Image tag the service runs. The deploy workflow pushes :latest and forces a new deployment."
  type        = string
  default     = "latest"
}
