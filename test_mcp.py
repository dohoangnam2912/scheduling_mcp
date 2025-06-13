from functools import wraps
import json
import logging
from datetime import datetime, date
from enum import Enum
from typing import Optional, Annotated, List, Dict
import uuid
import os

import aiohttp
from pydantic import BaseModel, Field, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import BearerAuthProvider
from fastmcp.server.auth.providers.bearer import RSAKeyPair
from fastmcp.server.dependencies import get_access_token, AccessToken


# --- 1. Centralized Configuration ---
class AppSettings(BaseSettings):
    """Manages application settings via environment variables or a .env file."""
    model_config = SettingsConfigDict(env_prefix='APP_')
    api_base_url: str = "https://main-sf6r.onrender.com"
    request_timeout: int = 15

settings = AppSettings()

# --- 2. Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("business-scheduler-mcp")

# --- 3. Improved Error Handling Decorator ---
def handle_tool_errors(func):
    """A decorator to standardize error handling and provide detailed validation feedback."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            ctx = kwargs.get('ctx')
            if ctx:
                await ctx.debug(f"Executing tool: {func.__name__}")
            return await func(*args, **kwargs)
        except ValidationError as e:
            first_error = e.errors()[0]
            field = " -> ".join(map(str, first_error['loc']))
            msg = first_error['msg']
            error_msg = f"Invalid input for field '{field}': {msg}"
            logger.error(f"Validation error in {func.__name__}: {error_msg}")
            if ctx:
                await ctx.error(f"Validation failed: {error_msg}")
            raise ToolError(error_msg)
        except ToolError:
            raise
        except Exception as e:
            error_msg = f"An unexpected error occurred in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if ctx:
                await ctx.error(f"Unexpected server error: {error_msg}")
            raise ToolError(error_msg)
    return wrapper

# --- 4. FastMCP Setup with Scope-Based Authentication ---
key_pair = RSAKeyPair.generate()

# REASON FOR IMPROVEMENT:
# We're now adding `required_scopes`. This enforces that any token used to
# communicate with this server MUST have the 'api:access' scope. This acts
# as a baseline authorization check for the entire server.
auth_provider = BearerAuthProvider(
    public_key=key_pair.public_key,
    required_scopes=["api:access"]
)

mcp = FastMCP(
    name="business-scheduling-assistant",
    instructions="""
    This is an intelligent assistant for managing business schedules.
    It can create, update, delete, and query schedules for private lessons, group classes, or block-off time.
    All operations require a valid JSON Web Token (JWT) with appropriate scopes for authentication.
    """,
    auth=auth_provider,
    mask_error_details=True,
    on_duplicate_tools="error"
)

# --- 5. Pydantic Models for Inputs and Outputs ---
class ScheduleType(str, Enum):
    PRIVATE_LESSON = "PRIVATE_LESSON"
    GROUP_CLASS = "GROUP_CLASS"
    BLOCK_OFF = "BLOCK_OFF"

class ScheduleLocationType(str, Enum):
    VIRTUAL = "VIRTUAL"
    IN_PERSON = "IN_PERSON"
    CUSTOM_ADDRESS = "CUSTOM_ADDRESS"

class RecurringMode(str, Enum):
    SINGLE_OCCURRENCE = "SINGLE_OCCURRENCE"
    THIS_AND_FUTURE = "THIS_AND_FUTURE"
    ENTIRE_SERIES = "ENTIRE_SERIES"

class RecurringUpdateData(BaseModel):
    mode: RecurringMode
    instanceOriginalStartTime: datetime

class _CreateScheduleInternal(BaseModel):
    title: str
    startTime: datetime
    endTime: datetime
    scheduleType: ScheduleType
    scheduleLocationType: ScheduleLocationType
    organizationId: str
    locationId: str
    studentId: Optional[str] = None
    groupClassId: Optional[str] = None
    meetingUrl: Optional[str] = None
    roomId: Optional[str] = None
    customAddress: Optional[str] = None
    description: Optional[str] = None
    instructorId: Optional[str] = None
    isRecurring: bool = False
    recurrenceRule: Optional[str] = None

    @model_validator(mode='after')
    def check_conditional_fields(self) -> '_CreateScheduleInternal':
        if self.scheduleType == ScheduleType.PRIVATE_LESSON and not self.studentId:
            raise ValueError("`studentId` is required for a PRIVATE_LESSON.")
        if self.scheduleType == ScheduleType.GROUP_CLASS and not self.groupClassId:
            raise ValueError("`groupClassId` is required for a GROUP_CLASS.")
        if self.scheduleLocationType == ScheduleLocationType.VIRTUAL and not self.meetingUrl:
            raise ValueError("`meetingUrl` is required for a VIRTUAL location.")
        if self.scheduleLocationType == ScheduleLocationType.IN_PERSON and not self.roomId:
            raise ValueError("`roomId` is required for an IN_PERSON location.")
        if self.scheduleLocationType == ScheduleLocationType.CUSTOM_ADDRESS and not self.customAddress:
            raise ValueError("`customAddress` is required for a CUSTOM_ADDRESS location.")
        return self

class ScheduleRecord(BaseModel):
    scheduleRecordId: str
    title: str
    startTime: datetime
    endTime: datetime

class ScheduleListResponse(BaseModel):
    data: List[ScheduleRecord]

class GenericSuccessResponse(BaseModel):
    success: bool
    message: str
    scheduleRecordId: Optional[str] = None

# --- 6. Tools with Granular Scope-Based Authorization ---
def require_scope(required_scope: str):
    """Helper function to check for a specific scope in the access token."""
    access_token: AccessToken = get_access_token()
    if required_scope not in access_token.scopes:
        raise ToolError(f"Permission denied: Requires scope '{required_scope}'.")

@mcp.tool(
    description="Creates a new schedule entry (e.g., lesson, class, or block-off time).",
    tags={"write", "schedule"},
    annotations={"readOnlyHint": False}
)
@handle_tool_errors
async def create_schedule(
    # Arguments remain the same...
    title: Annotated[str, Field(description="Title or name of the schedule entry.", min_length=2)],
    startTime: Annotated[datetime, Field(description="The start time for the event.")],
    endTime: Annotated[datetime, Field(description="The end time for the event.")],
    scheduleType: Annotated[ScheduleType, Field(description="The type of schedule.")],
    scheduleLocationType: Annotated[ScheduleLocationType, Field(description="The location type.")],
    organizationId: Annotated[str, Field(description="The user's organization ID.")],
    locationId: Annotated[str, Field(description="The user's selected location ID.")],
    studentId: Optional[str] = None, groupClassId: Optional[str] = None, meetingUrl: Optional[str] = None,
    roomId: Optional[str] = None, customAddress: Optional[str] = None, description: Optional[str] = None,
    instructorId: Optional[str] = None, isRecurring: bool = False, recurrenceRule: Optional[str] = None,
    ctx: Context = None
) -> GenericSuccessResponse:
    # REASON FOR IMPROVEMENT:
    # We now explicitly check for the 'schedule:write' scope before proceeding.
    # This ensures only clients with write permissions can create schedules.
    require_scope("schedule:write")
    
    await ctx.info(f"Permissions verified. Creating schedule '{title}'...")
    
    headers = {"Authorization": f"Bearer {get_access_token().token}"}
    schedule_data = _CreateScheduleInternal(**locals())
    payload = schedule_data.model_dump(mode='json', exclude_none=True)
    
    await ctx.info("Sending request to create schedule in backend...")
    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.post("/api/v1/account/schedule", headers=headers, json=payload) as response:
            if not response.ok:
                error_detail = await response.text()
                await ctx.error(f"Backend API failed with status {response.status}.")
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()
    
    schedule_id = result.get('scheduleRecordId')
    logger.info(f"Successfully created schedule '{title}', ID: {schedule_id}")
    await ctx.info("Successfully created schedule.")
    return GenericSuccessResponse(
        success=True, message=f"Successfully created schedule '{title}'.", scheduleRecordId=schedule_id
    )

@mcp.tool(
    description="Deletes a schedule entry. Can delete a single instance or an entire series.",
    tags={"write", "schedule"},
    annotations={"readOnlyHint": False, "destructiveHint": True}
)
@handle_tool_errors
async def delete_schedule(
    scheduleRecordId: Annotated[str, Field(description="The unique ID of the schedule entry to delete.")],
    recurring_data: Annotated[Optional[RecurringUpdateData], Field(description="Required ONLY if deleting an instance of a recurring event.")] = None,
    ctx: Context = None
) -> GenericSuccessResponse:
    require_scope("schedule:write")
    await ctx.info(f"Permissions verified. Deleting schedule ID: {scheduleRecordId}...")
    
    headers = {"Authorization": f"Bearer {get_access_token().token}"}
    payload = {"scheduleRecordId": scheduleRecordId}
    if recurring_data:
        payload.update(recurring_data.model_dump())

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.delete("/api/v1/account/schedule", headers=headers, json=payload) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
    
    await ctx.info(f"Successfully deleted schedule ID: {scheduleRecordId}.")
    return GenericSuccessResponse(
        success=True, message=f"Successfully deleted schedule ID {scheduleRecordId}.", scheduleRecordId=scheduleRecordId
    )

@mcp.tool(
    description="Updates the start and end time of an existing schedule entry.",
    tags={"write", "schedule"},
    annotations={"readOnlyHint": False}
)
@handle_tool_errors
async def update_schedule(
    scheduleRecordId: Annotated[str, Field(description="ID of the schedule to update.")],
    newStartTime: Annotated[datetime, Field(description="The new start time for the event.")],
    newEndTime: Annotated[datetime, Field(description="The new end time for the event.")],
    recurring_data: Annotated[Optional[RecurringUpdateData], Field(description="Required ONLY if updating an instance of a recurring event.")] = None,
    ctx: Context = None
) -> GenericSuccessResponse:
    require_scope("schedule:write")
    await ctx.info(f"Permissions verified. Updating schedule ID: {scheduleRecordId}...")
    
    headers = {"Authorization": f"Bearer {get_access_token().token}"}
    payload = {"startTime": newStartTime.isoformat(), "endTime": newEndTime.isoformat()}
    if recurring_data:
        payload.update(recurring_data.model_dump())

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.put(f"/api/v1/account/schedule/{scheduleRecordId}", headers=headers, json=payload) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
        
    await ctx.info(f"Successfully updated schedule ID: {scheduleRecordId}.")
    return GenericSuccessResponse(
        success=True, message=f"Successfully updated schedule ID {scheduleRecordId}.", scheduleRecordId=scheduleRecordId
    )

@mcp.tool(
    description="Queries for schedules within a given date range.",
    tags={"read", "schedule"},
    annotations={"readOnlyHint": True}
)
@handle_tool_errors
async def query_schedules(
    startDate: Annotated[date, Field(description="The start date for the query range.")],
    endDate: Annotated[date, Field(description="The end date for the query range.")],
    organizationId: Annotated[str, Field(description="The user's organization ID.")],
    locationId: Annotated[str, Field(description="The user's selected location ID.")],
    ctx: Context = None
) -> ScheduleListResponse:
    # This tool requires the 'schedule:read' scope.
    require_scope("schedule:read")
    await ctx.info(f"Permissions verified. Querying schedules from {startDate} to {endDate}...")
    
    headers = {"Authorization": f"Bearer {get_access_token().token}"}
    query_params = {
        "startDate": startDate.isoformat(), "endDate": endDate.isoformat(),
        "organizationId": organizationId, "locationId": locationId,
    }

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.get("/api/v1/account/schedule", headers=headers, params=query_params) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()

    count = len(result.get('data', []))
    await ctx.info(f"Found {count} schedules.")
    return ScheduleListResponse(**result)

# --- 7. Improved Prompts with Input Validation ---
@mcp.prompt(description="Formats a confirmation for a newly created schedule.")
def format_creation_confirmation(title: str, startTime: datetime, scheduleRecordId: str) -> str:
    friendly_time = startTime.strftime("%A, %B %d, %Y at %I:%M %p")
    return f"✅ Success! The schedule '{title}' has been created for {friendly_time}. The new Schedule ID is `{scheduleRecordId}`."

@mcp.prompt(description="Summarizes a list of schedules into a readable format.")
def summarize_schedule_list(schedule_data: ScheduleListResponse) -> str:
    schedules = schedule_data.data
    if not schedules:
        return "No schedules were found in the given timeframe."
    
    lines = [f"Found {len(schedules)} schedule(s):"]
    schedules.sort(key=lambda s: s.startTime)

    for item in schedules:
        friendly_time = item.startTime.strftime('%a, %b %d at %I:%M %p')
        lines.append(f"- **{item.title}** on {friendly_time} (ID: `{item.scheduleRecordId}`)")
        
    return "\n".join(lines)

# --- 8. Server Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # REASON FOR IMPROVEMENT:
    # The development token now includes the specific scopes needed to pass our
    # new, more granular authorization checks.
    sample_token = key_pair.create_token(
        subject="dev-user",
        scopes=["api:access", "schedule:read", "schedule:write"]
    )
    logger.info("="*80)
    logger.info("🚀 Starting Business Scheduling Assistant MCP")
    logger.info(f"🔑 Development JWT (use with 'Bearer' scheme): {sample_token}")
    logger.info("="*80)

    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
