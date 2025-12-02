"""
CloudWatch logging utilities for sending evaluation results directly to CloudWatch Logs.

NOTE: This module is temporary and will be removed in the future once ADOT integration is complete.
"""

import json
import logging
import os
import time

import boto3

logger = logging.getLogger(__name__)

# Module-level CloudWatch client (lazy initialization)
_cloudwatch_client = None


def _get_cloudwatch_client():
    """
    Get or create the CloudWatch Logs client (singleton pattern).

    Returns:
        boto3 CloudWatch Logs client
    """
    global _cloudwatch_client
    if _cloudwatch_client is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        _cloudwatch_client = boto3.client("logs", region_name=region)
    return _cloudwatch_client


def _parse_log_config_from_env(config_id: str):
    """
    Parse log group and stream from environment variables.

    Args:
        config_id: The config ID (not used for log group, only for ARN)

    Returns:
        Tuple of (destination_log_group, log_stream_name, service_name, resource_log_group)
        - destination_log_group: From EVALUATION_RESULTS_LOG_GROUP env var (actual destination where logs are sent)
        - resource_log_group: The log group from OTEL_RESOURCE_ATTRIBUTES for resource attributes in EMF
    """
    # Parse from OTEL_EXPORTER_OTLP_LOGS_HEADERS
    logs_headers = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS", "")
    log_stream = "default"

    if logs_headers:
        for header in logs_headers.split(","):
            if "=" in header:
                key, value = header.split("=", 1)
                if key.strip() == "x-aws-log-stream":
                    log_stream = value.strip()

    # Destination log group is from EVALUATION_RESULTS_LOG_GROUP environment variable
    destination_log_group = os.environ.get("EVALUATION_RESULTS_LOG_GROUP", "default_strands_evals_results")
    destination_log_group = f"/aws/bedrock-agentcore/evaluations/results/{destination_log_group}"

    # Get resource log group from OTEL_RESOURCE_ATTRIBUTES (for EMF resource attributes)
    resource_log_group = None
    resource_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    for attr in resource_attrs.split(","):
        if "=" in attr:
            key, value = attr.split("=", 1)
            if key.strip() == "aws.log.group.names":
                resource_log_group = value.strip()

    # Get service name from OTEL_RESOURCE_ATTRIBUTES
    service_name = None
    for attr in resource_attrs.split(","):
        if "=" in attr:
            key, value = attr.split("=", 1)
            if key.strip() == "service.name":
                service_name = value.strip()
                break

    if not service_name:
        raise ValueError("service.name must be set in OTEL_RESOURCE_ATTRIBUTES environment variable")

    return destination_log_group, log_stream, service_name, resource_log_group


def _send_to_cloudwatch(
    message: str,
    log_data: dict,
    trace_id: str,
    evaluator_name: str,
    score: float,
    config_id: str,
    label: str = "",
):
    """
    Send log event directly to CloudWatch Logs using boto3 in EMF format.

    Args:
        message: The log message
        log_data: Dictionary containing the log event data
        trace_id: The OpenTelemetry trace ID
        evaluator_name: The name of the evaluator (e.g., "Custom.HelpfulnessEvaluator")
        score: The evaluation score
        config_id: The config ID for log group path
        label: The evaluation label (optional)
    """
    try:
        destination_log_group, log_stream, service_name, resource_log_group = _parse_log_config_from_env(config_id)
        if not destination_log_group:
            logger.warning("No destination log group configured, skipping CloudWatch logging")
            return

        # Get the singleton CloudWatch client
        cloudwatch_client = _get_cloudwatch_client()

        # Ensure destination log group exists
        try:
            cloudwatch_client.create_log_group(logGroupName=destination_log_group)
        except cloudwatch_client.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            logger.warning(f"Failed to create log group: {str(e)}")

        # Ensure log stream exists
        try:
            cloudwatch_client.create_log_stream(logGroupName=destination_log_group, logStreamName=log_stream)
        except cloudwatch_client.exceptions.ResourceAlreadyExistsException:
            pass
        except Exception as e:
            logger.warning(f"Failed to create log stream: {str(e)}")

        # Get sequence token for the log stream
        sequence_token = None
        try:
            response = cloudwatch_client.describe_log_streams(
                logGroupName=destination_log_group, logStreamNamePrefix=log_stream
            )
            if response["logStreams"]:
                sequence_token = response["logStreams"][0].get("uploadSequenceToken")
        except Exception as e:
            logger.warning(f"Failed to get sequence token: {str(e)}")

        # Get current timestamp
        current_time_ns = time.time_ns()
        current_time_ms = int(current_time_ns / 1_000_000)

        # Extract online evaluation config ID from ARN
        online_eval_config_id = ""
        if "aws.bedrock_agentcore.online_evaluation_config.arn" in log_data:
            arn = log_data["aws.bedrock_agentcore.online_evaluation_config.arn"]
            if "/" in arn:
                online_eval_config_id = arn.split("/")[-1]

        emf_log = {
            "resource": {
                "attributes": {
                    "aws.local.service": service_name,
                    "aws.service.type": "gen_ai_agent",
                    "telemetry.sdk.language": "python",
                    "service.name": service_name,
                    "aws.log.group.names": resource_log_group if resource_log_group else destination_log_group,
                }
            },
            "timeUnixNano": current_time_ns,
            "observedTimeUnixNano": current_time_ns,
            "severityText": "INFO",
            "body": message,
            "attributes": {
                "otelServiceName": service_name,
                "otelTraceID": trace_id,
                "otelTraceSampled": True,
                **log_data,
            },
            "flags": 1,
            "traceId": trace_id,
            "onlineEvaluationConfigId": online_eval_config_id,
            evaluator_name: score,
        }

        # Add label if provided
        emf_log["label"] = label or "YES"
        emf_log["service.name"] = service_name

        # Add EMF metadata for CloudWatch metrics
        emf_log["_aws"] = {
            "Timestamp": current_time_ms,
            "CloudWatchMetrics": [
                {
                    "Namespace": "Bedrock-AgentCore/Evaluations",
                    "Dimensions": [
                        ["service.name"],
                        ["label", "service.name"],
                        ["service.name", "onlineEvaluationConfigId"],
                        ["label", "service.name", "onlineEvaluationConfigId"],
                    ],
                    "Metrics": [{"Name": evaluator_name, "Unit": "None"}],
                }
            ],
        }

        log_event = {"timestamp": current_time_ms, "message": json.dumps(emf_log)}
        put_log_params = {"logGroupName": destination_log_group, "logStreamName": log_stream, "logEvents": [log_event]}

        if sequence_token:
            put_log_params["sequenceToken"] = sequence_token

        cloudwatch_client.put_log_events(**put_log_params)

    except Exception as e:
        logger.warning(f"Failed to send log to CloudWatch: {str(e)}")
