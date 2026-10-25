output "url" {
  description = "Public HTTPS address of the demo (page + API)."
  value       = "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "api_key" {
  description = "Access key for /ask and /agent (X-API-Key header, or the field on the demo page)."
  value       = random_password.api_key.result
  sensitive   = true
}

output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "secret_parameters" {
  description = "Set these once with `aws ssm put-parameter --overwrite` (see infra/README.md)."
  value       = { for k, p in aws_ssm_parameter.external : k => p.name }
}

# Values for the GitHub repository variables used by .github/workflows/deploy.yml.
output "github_variables" {
  value = {
    AWS_REGION                 = var.aws_region
    AWS_DEPLOY_ROLE_ARN        = aws_iam_role.github_deploy.arn
    ECR_REPOSITORY             = aws_ecr_repository.api.repository_url
    ECS_CLUSTER                = aws_ecs_cluster.main.name
    ECS_SERVICE                = aws_ecs_service.api.name
    WEB_BUCKET                 = aws_s3_bucket.web.id
    CLOUDFRONT_DISTRIBUTION_ID = aws_cloudfront_distribution.main.id
  }
}
