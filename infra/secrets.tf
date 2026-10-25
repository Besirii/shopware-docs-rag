# Secrets live in SSM Parameter Store (SecureString, free tier) and reach the container as
# environment variables at start. The third-party keys are NOT in Terraform: Terraform creates
# the parameters with a placeholder and never touches the value again, you set the real value
# once with the AWS CLI (see infra/README.md). So they never end up in code or in the state file.

locals {
  external_secrets = {
    LLM_API_KEY    = "Anthropic API key (sk-ant-api03-...)"
    QDRANT_API_KEY = "Qdrant Cloud API key"
  }
}

resource "aws_ssm_parameter" "external" {
  for_each = local.external_secrets

  name        = "/${var.project}/${each.key}"
  description = each.value
  type        = "SecureString"
  value       = "SET_ME"

  lifecycle {
    ignore_changes = [value]
  }
}

# The access key for /ask and /agent is generated here: it only protects this demo, and you can
# read it with `terraform output -raw api_key` to share it with whoever should try the demo.
resource "random_password" "api_key" {
  length  = 32
  special = false
}

resource "aws_ssm_parameter" "api_key" {
  name        = "/${var.project}/API_KEY"
  description = "Access key required in the X-API-Key header for /ask and /agent"
  type        = "SecureString"
  value       = random_password.api_key.result
}

# Secret header CloudFront adds to every request to the load balancer. The load balancer only
# forwards requests that carry it, so nobody can bypass CloudFront by calling the ALB directly.
resource "random_password" "origin_verify" {
  length  = 40
  special = false
}

locals {
  container_secrets = merge(
    { for k, p in aws_ssm_parameter.external : k => p.arn },
    { API_KEY = aws_ssm_parameter.api_key.arn },
  )
}
