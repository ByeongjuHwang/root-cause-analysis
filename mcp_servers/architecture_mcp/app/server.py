import sys
import logging
from mcp.server.fastmcp import FastMCP

from .repository import (
    get_system_diagram,
    get_service_catalog,
    get_service_metadata,
    get_service_dependencies,
    get_related_services,
    find_path,
    infer_blast_radius,
)

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

mcp = FastMCP("architecture-mcp")


@mcp.resource("arch://system/latest", mime_type="application/json")
def system_diagram_resource() -> dict:
    """Return the latest architecture diagram metadata and topology graph."""
    return get_system_diagram()


@mcp.resource("arch://services/catalog", mime_type="application/json")
def service_catalog_resource() -> dict:
    """Return the full service catalog."""
    return get_service_catalog()


@mcp.tool()
def get_service_metadata_tool(service: str) -> dict:
    """Return metadata for a target service."""
    logging.info("get_service_metadata called: service=%s", service)
    return get_service_metadata(service)


@mcp.tool()
def get_service_dependencies_tool(service: str) -> dict:
    """Return dependency relationships for a target service."""
    logging.info("get_service_dependencies called: service=%s", service)
    return get_service_dependencies(service)


@mcp.tool()
def get_related_services_tool(service: str) -> dict:
    """Return directly related services for a target service."""
    logging.info("get_related_services called: service=%s", service)
    return get_related_services(service)


@mcp.tool()
def find_path_tool(source: str, target: str) -> dict:
    """Find a directed dependency path between two services."""
    logging.info("find_path called: source=%s target=%s", source, target)
    return find_path(source, target)


@mcp.tool()
def infer_blast_radius_tool(service: str, depth: int = 2) -> dict:
    """Infer likely upstream impact radius from a failing service."""
    logging.info("infer_blast_radius called: service=%s depth=%s", service, depth)
    return infer_blast_radius(service, depth)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()    