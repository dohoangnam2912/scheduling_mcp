from functools import wraps
import json
import logging
from datetime import datetime, date
from enum import Enum
from typing import Optional, Annotated, List, Dict
import uuid
import os

import aiohttp
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError

# --- 1. Centralized Configuration ---
class AppSettings(BaseSettings):
    """Manages application settings via environment variables or a .env file."""
    model_config = SettingsConfigDict(env_prefix='APP_')
    api_base_url: str = "http://localhost:8000"
    request_timeout: int = 15

settings = AppSettings()

# --- 2. Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("business-scheduler-mcp")

# --- 3. Error Handling Decorator (No longer needed for API errors) ---
# The logic is now handled inside each tool for better error detail.
def handle_tool_errors(func):
    """A decorator to standardize non-API error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            error_msg = f"Invalid input: {e}"
            logger.error(f"Validation error in {func.__name__}: {error_msg}")
            raise ToolError(error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred in {func.__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ToolError(error_msg)
    return wrapper

# --- 4. FastMCP Setup ---
mcp = FastMCP(
    name="business-scheduling-assistant",
    instructions="""
    This is an intelligent assistant for managing business schedules.
    It can create, update, delete, and query schedules for private lessons, group classes, or block-off time.
    All operations require a valid JSON Web Token (JWT) for authentication.
    """,
    mask_error_details=False,
    on_duplicate_tools="error"
)

# --- 5. Pydantic Models with Business Logic ---
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

class CreateScheduleData(BaseModel):
    title: str = Field(..., description="Title or name of the schedule entry.")
    startTime: datetime
    endTime: datetime
    scheduleType: ScheduleType = Field(..., 
        description="The type of schedule. Ask the user if it's for a 'private lesson', a 'group class', or to 'block off' time. Use the corresponding technical value: PRIVATE_LESSON, GROUP_CLASS, or BLOCK_OFF."
    )
    scheduleLocationType: ScheduleLocationType = Field(...,
        description="The location type for the schedule. Ask the user if it will be 'virtual', 'in-person' at the studio, or at a 'custom address'. Use the corresponding technical value: VIRTUAL, IN_PERSON, or CUSTOM_ADDRESS."
    )
    organizationId: str = Field(..., description="The user's organization ID.")
    locationId: str = Field(..., description="The user's selected location ID.")
    studentId: Optional[str] = Field(None, description="Required if scheduleType is PRIVATE_LESSON.")
    groupClassId: Optional[str] = Field(None, description="Required if scheduleType is GROUP_CLASS.")
    meetingUrl: Optional[str] = Field(None, description="Required if scheduleLocationType is VIRTUAL.")
    roomId: Optional[str] = Field(None, description="Required if scheduleLocationType is IN_PERSON.")
    customAddress: Optional[str] = Field(None, description="Required if scheduleLocationType is CUSTOM_ADDRESS.")
    description: Optional[str] = None
    instructorId: Optional[str] = None
    isRecurring: bool = False
    recurrenceRule: Optional[str] = Field(None, description="iCalendar RRULE string (e.g., 'FREQ=WEEKLY;BYDAY=MO').")
    
    @model_validator(mode='after')
    def check_conditional_fields(self) -> 'CreateScheduleData':
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

class UpdateScheduleBody(BaseModel):
    """Body for the update request. Only handles time updates."""
    startTime: datetime = Field(..., description="The new start time.")
    endTime: datetime = Field(..., description="The new end time.")
    
    mode: Optional[RecurringMode] = Field(None, description="Required if updating a recurring event.")
    instanceOriginalStartTime: Optional[datetime] = Field(None, description="For recurring events, the original start time of the instance being edited.")

class DeleteScheduleBody(BaseModel):
    scheduleRecordId: str
    mode: Optional[RecurringMode] = Field(None, description="Required if deleting a recurring event.")
    instanceOriginalStartTime: Optional[datetime] = Field(None, description="For recurring events, the original start time of the instance being deleted.")

class QueryScheduleParams(BaseModel):
    startDate: date
    endDate: date
    organizationId: str
    locationId: str

# --- 6. Corrected Tools with Improved Error Handling ---
@mcp.tool(description="Creates a new schedule entry (e.g., lesson, class, or block-off).")
@handle_tool_errors
async def create_schedule(jwt: Annotated[str, Field(description="User's JWT for authentication.")], data: CreateScheduleData, ctx: Context) -> dict:
    logger.info(f"Attempting to create schedule: {data.title}")
    headers = {"Authorization": f"Bearer {jwt}"}
    payload = data.model_dump(mode='json', exclude_none=True)

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.post("/api/v1/account/schedule", headers=headers, json=payload) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()
    
    logger.info(f"Successfully created schedule '{data.title}', ID: {result.get('scheduleRecordId')}")
    return result

@mcp.tool(description="Deletes a schedule entry.")
@handle_tool_errors
async def delete_schedule(jwt: Annotated[str, Field(description="User's JWT for authentication.")], data: DeleteScheduleBody, ctx: Context) -> dict:
    logger.info(f"Attempting to delete schedule ID: {data.scheduleRecordId}")
    headers = {"Authorization": f"Bearer {jwt}"}
    
    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.delete("/api/v1/account/schedule", headers=headers, json=data.model_dump(mode='json', exclude_none=True)) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()
        
    logger.info(f"Successfully deleted schedule ID: {data.scheduleRecordId}")
    return result

@mcp.tool(description="Updates the start and end time of an existing schedule entry.")
@handle_tool_errors
async def update_schedule(
    jwt: Annotated[str, Field(description="User's JWT for authentication.")],
    scheduleRecordId: Annotated[str, Field(description="ID of the schedule to update.")],
    data: UpdateScheduleBody,
    ctx: Context
) -> dict:
    logger.info(f"Attempting to update schedule ID: {scheduleRecordId}")
    headers = {"Authorization": f"Bearer {jwt}"}

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.put(f"/api/v1/account/schedule/{scheduleRecordId}", headers=headers, json=data.model_dump(mode='json', exclude_none=True)) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()
        
    logger.info(f"Successfully updated schedule ID: {scheduleRecordId}")
    return result

@mcp.tool(description="Queries for schedules within a given date range.")
@handle_tool_errors
async def query_schedules(jwt: Annotated[str, Field(description="User's JWT for authentication.")], params: QueryScheduleParams, ctx: Context) -> dict:
    logger.info(f"Querying schedules from {params.startDate} to {params.endDate}")
    headers = {"Authorization": f"Bearer {jwt}"}
    query_params = {
        "startDate": params.startDate.isoformat(),
        "endDate": params.endDate.isoformat(),
        "organizationId": params.organizationId,
        "locationId": params.locationId,
    }

    async with aiohttp.ClientSession(base_url=settings.api_base_url, timeout=aiohttp.ClientTimeout(total=settings.request_timeout)) as session:
        async with session.get("/api/v1/account/schedule", headers=headers, params=query_params) as response:
            if not response.ok:
                error_detail = await response.text()
                raise ToolError(f"API Error ({response.status}): {error_detail}")
            result = await response.json()

    logger.info(f"Found {len(result.get('data', []))} schedules in the specified range.")
    return result

# --- 7. Prompts ---
@mcp.prompt(description="Formats a confirmation for a newly created schedule.")
def format_creation_confirmation(title: str, startTime: datetime, scheduleRecordId: str) -> str:
    friendly_time = startTime.strftime("%A, %B %d, %Y at %I:%M %p")
    return f"✅ Success! The schedule '{title}' has been created for {friendly_time}. The new Schedule ID is `{scheduleRecordId}`."

@mcp.prompt(description="Summarizes a list of schedules into a readable format.")
def summarize_schedule_list(schedule_data: Dict[str, List]) -> str:
    schedules = schedule_data.get('data', [])
    if not schedules:
        return "No schedules were found in the given timeframe."
    
    lines = [f"Found {len(schedules)} schedule(s):"]
    schedules.sort(key=lambda s: s.get('startTime', ''))

    for item in schedules:
        start_time = datetime.fromisoformat(item['startTime'].replace('Z', '+00:00'))
        friendly_time = start_time.strftime('%a, %b %d at %I:%M %p')
        lines.append(f"- **{item['title']}** on {friendly_time} (ID: `{item.get('scheduleRecordId', 'N/A')}`)")
        
    return "\n".join(lines)

# --- 8. Server Execution ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use Render's dynamic port
    logger.info(f"Starting Business Scheduling Assistant MCP at http://0.0.0.0:{port}")
    mcp.run(transport="sse", host="0.0.0.0", port=port)