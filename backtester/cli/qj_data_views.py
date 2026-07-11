"""
qj_data_views — Rich views for qj-data
--------------------------------------

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from backtester.cli.qj_data_api import QJDataSnapshot

console = Console()


def clear_screen() -> None:
    console.clear()


def _fmt_bool(value: Any) -> str:
    return "[bold green]yes[/bold green]" if bool(value) else "[bold red]no[/bold red]"


def _fmt_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value) if value is not None else "-"


def _make_title_text(title: str) -> Text:
    text = Text()
    text.append(title, style="bold")
    return text


def _make_info_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=False,
        expand=True,
        padding=(0, 1),
    )
    table.add_column(style="bold bright_cyan", no_wrap=True)
    table.add_column(style="white")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _make_list_panel(title: str, items: list[str], *, border_style: str) -> Panel:
    content = "\n".join(f"• {item}" for item in items) if items else "[dim]No items available[/dim]"
    return Panel(
        content,
        title=_make_title_text(title),
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 1),
    )


def _fmt_status(text: str) -> str:
    return f"[bold bright_yellow]{text}[/bold bright_yellow]"


def show_home_banner(_snapshot: QJDataSnapshot) -> None:
    ascii_title = Text(
        "\n".join(
            [
                " ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗ ",
                "██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝ ",
                "██║   ██║██║   ██║███████║██╔██╗ ██║   ██║    ",
                "██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║    ",
                "╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║    ",
                " ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ",
                "",
                "     ██╗ ██████╗ ██╗   ██╗██████╗ ███╗   ██╗███████╗██╗   ██╗ ",
                "     ██║██╔═══██╗██║   ██║██╔══██╗████╗  ██║██╔════╝╚██╗ ██╔╝ ",
                "     ██║██║   ██║██║   ██║██████╔╝██╔██╗ ██║█████╗   ╚████╔╝  ",
                "██   ██║██║   ██║██║   ██║██╔══██╗██║╚██╗██║██╔══╝    ╚██╔╝   ",
                "╚█████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║███████╗   ██║    ",
            ]
        ),
        style="bold bright_cyan",
        justify="center",
    )
    helper = Text("HELPER EXPLORER", style="bold bright_magenta", justify="center")

    body = Table.grid(padding=(0, 1))
    body.add_row(ascii_title)
    body.add_row(Text(""))
    body.add_row(helper)

    console.print(Rule(style="bright_cyan"))
    console.print(
        Panel(
            Align.center(body),
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2),
        )
    )
    console.print(Rule(style="bright_cyan"))


def show_overview(snapshot: QJDataSnapshot) -> None:
    data_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_green]Available Data[/bold bright_green]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
        expand=True,
    )
    data_table.add_column("Available Data Sources", style="bold bright_cyan")
    data_table.add_column("Value", style="bold bright_white", justify="right")
    data_table.add_row("Asset Classes", str(len(snapshot.asset_classes)))
    data_table.add_row("Sources", str(len(snapshot.sources)))
    data_table.add_row("Granularity Count", str(len(snapshot.granularities)))
    data_table.add_row(
        "Available Granularities",
        _fmt_list([item.get("id", "-") for item in snapshot.granularities]),
    )
    data_table.add_row("Datasets", str(len(snapshot.datasets)))
    data_table.add_row("Symbols", str(len(snapshot.available_symbols)))
    data_table.add_row("Universes", str(len(snapshot.example_universes)))
    console.print(data_table)

    strategy_summary = snapshot.catalog_doc.get("strategy_summary", {})
    strategy_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_magenta]Strategies[/bold bright_magenta]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
        expand=True,
    )
    strategy_table.add_column("Strategies", style="bold bright_cyan")
    strategy_table.add_column("Count", style="bold bright_white", justify="right")
    for key, label in [
        ("total", "Total Number of Strategies"),
        ("weights", "Type Strategy Weight"),
        ("orders", "Type Strategy Orders"),
        ("walk_forward", "Type Strategy Walk Forward"),
        ("cacheable", "Cacheable Strategies"),
    ]:
        if key in strategy_summary:
            strategy_table.add_row(label, str(strategy_summary[key]))

    console.print(strategy_table)


def show_view_all(snapshot: QJDataSnapshot) -> None:
    top_data = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_green]Data Availability[/bold bright_green]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
        expand=True,
    )
    top_data.add_column("Category", style="bold bright_cyan")
    top_data.add_column("Value", style="bold bright_white", overflow="fold")
    top_data.add_row("Asset Classes", _fmt_list([str(item) for item in snapshot.asset_classes]))
    top_data.add_row(
        "Sources",
        _fmt_list([str(item.get("id", "-")) for item in snapshot.sources]),
    )
    top_data.add_row(
        "Granularities",
        _fmt_list([str(item.get("id", "-")) for item in snapshot.granularities]),
    )
    top_data.add_row("Datasets", str(len(snapshot.datasets)))
    top_data.add_row("Symbols", str(len(snapshot.available_symbols)))
    top_data.add_row("Universes", str(len(snapshot.example_universes)))

    strategy_summary = snapshot.catalog_doc.get("strategy_summary", {})
    strategies = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_magenta]Strategy Types[/bold bright_magenta]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
        expand=True,
    )
    strategies.add_column("Strategy Metric", style="bold bright_cyan")
    strategies.add_column("Count", style="bold bright_white", justify="right")
    for key, label in [
        ("total", "Total Number of Strategies"),
        ("weights", "Weight Strategies"),
        ("orders", "Order Strategies"),
        ("walk_forward", "Walk-Forward Strategies"),
        ("cacheable", "Cacheable Strategies"),
        ("cache_scope_global", "Global Cache Scope"),
        ("cache_scope_tenant", "Tenant Cache Scope"),
    ]:
        if key in strategy_summary:
            strategies.add_row(label, str(strategy_summary[key]))

    console.print(Columns([top_data, strategies], equal=True, expand=True))

    sources_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_green]Source Coverage[/bold bright_green]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    sources_table.add_column("Source", style="bold bright_green", no_wrap=True)
    sources_table.add_column("Label", style="bold bright_white")
    sources_table.add_column("Prepare Needs Key", justify="center")
    sources_table.add_column("Developer Notes", overflow="fold")
    for source in snapshot.sources:
        supported = sorted(
            item.get("id", "-")
            for item in snapshot.granularities
            if source.get("id") in item.get("supported_sources", [])
        )
        notes = str(source.get("notes", "-"))
        if supported:
            notes = f"Granularities: {', '.join(supported)}"
            source_notes = str(source.get("notes", "")).strip()
            if source_notes and source_notes != "-":
                notes = f"{notes}. {source_notes}"
        sources_table.add_row(
            str(source.get("id", "-")),
            str(source.get("label", "-")),
            _fmt_bool(source.get("requires_api_key_for_prepare", True)),
            notes,
        )
    console.print(sources_table)

    granularity_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_magenta]Granularity Coverage[/bold bright_magenta]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    granularity_table.add_column("Granularity", style="bold bright_cyan", no_wrap=True)
    granularity_table.add_column("Category", style="bright_yellow")
    granularity_table.add_column("Best For", style="bold bright_white")
    granularity_table.add_column("Supported Sources", overflow="fold")
    for granularity in snapshot.granularities:
        granularity_id = str(granularity.get("id", "-"))
        if granularity_id == "1d":
            best_for = "Daily systems, portfolio rotation, end-of-day research"
        elif granularity_id in {"1h", "30m"}:
            best_for = "Medium-frequency trading and intraday trend systems"
        else:
            best_for = "Higher-frequency intraday monitoring and execution timing"
        granularity_table.add_row(
            granularity_id,
            str(granularity.get("category", "-")),
            best_for,
            _fmt_list(granularity.get("supported_sources")),
        )
    console.print(granularity_table)

    datasets_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_yellow]Dataset & Endpoint Coverage[/bold bright_yellow]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    datasets_table.add_column("Dataset", style="bold bright_cyan", no_wrap=True)
    datasets_table.add_column("Label", style="bold bright_white")
    datasets_table.add_column("Endpoint", overflow="fold")
    datasets_table.add_column("Needs Key", justify="center")
    for dataset in snapshot.datasets:
        datasets_table.add_row(
            str(dataset.get("id", "-")),
            str(dataset.get("label", "-")),
            str(dataset.get("payload_endpoint", "-")),
            _fmt_bool(dataset.get("payload_requires_api_key", True)),
        )
    console.print(datasets_table)

    universes_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_red]Universe Coverage[/bold bright_red]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    universes_table.add_column("Universe", style="bold bright_cyan", no_wrap=True)
    universes_table.add_column("Symbols", justify="right", style="bright_yellow")
    universes_table.add_column("Examples", overflow="fold")
    for universe in snapshot.example_universes:
        symbols = [str(symbol) for symbol in universe.get("symbols", [])]
        universes_table.add_row(
            str(universe.get("label", "-")),
            str(len(symbols)),
            _fmt_list(symbols[:6]) if len(symbols) > 6 else _fmt_list(symbols),
        )
    console.print(universes_table)

    symbols_table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_cyan]Symbol Availability Snapshot[/bold bright_cyan]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    symbols_table.add_column("Symbol", style="bold bright_green", no_wrap=True)
    symbols_table.add_column("Universe", overflow="fold")
    symbols_table.add_column("Asset Class", overflow="fold")
    symbols_table.add_column("Granularities", overflow="fold")
    symbols_table.add_column("Period / Dates", overflow="fold")
    for symbol_item in snapshot.available_symbols:
        symbols_table.add_row(
            str(symbol_item.get("symbol", "-")),
            _fmt_list(symbol_item.get("universe_labels")),
            "[dim]Not exposed publicly[/dim]",
            "[dim]Per-symbol coverage not exposed[/dim]",
            "[dim]Per-symbol period/date range not exposed[/dim]",
        )
    console.print(symbols_table)


def show_sources(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_green]Sources[/bold bright_green]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("ID", style="bright_cyan", no_wrap=True)
    table.add_column("Label", style="bold bright_white")
    table.add_column("Help visible", justify="center")
    table.add_column("Prepare needs key", justify="center")
    table.add_column("Notes", overflow="fold")

    for source in snapshot.sources:
        table.add_row(
            str(source.get("id", "-")),
            str(source.get("label", "-")),
            _fmt_bool(source.get("public_help_visible", False)),
            _fmt_bool(source.get("requires_api_key_for_prepare", True)),
            str(source.get("notes", "-")),
        )

    console.print(table)


def show_granularities(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_magenta]Granularities[/bold bright_magenta]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("ID", style="bright_cyan", no_wrap=True)
    table.add_column("Category", style="bright_yellow")
    table.add_column("Label", style="bold bright_white")
    table.add_column("Supported sources", overflow="fold")

    for granularity in snapshot.granularities:
        table.add_row(
            str(granularity.get("id", "-")),
            str(granularity.get("category", "-")),
            str(granularity.get("label", "-")),
            _fmt_list(granularity.get("supported_sources")),
        )

    console.print(table)


def show_asset_classes(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_blue]Asset Classes[/bold bright_blue]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("#", style="bright_cyan", no_wrap=True, justify="right")
    table.add_column("Asset class", style="bold bright_white")

    for idx, asset_class in enumerate(snapshot.asset_classes, start=1):
        table.add_row(str(idx), str(asset_class))

    console.print(table)


def show_datasets(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_yellow]Datasets[/bold bright_yellow]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("ID", style="bright_cyan", no_wrap=True)
    table.add_column("Label", style="bold bright_white")
    table.add_column("Endpoint", overflow="fold")
    table.add_column("Needs key", justify="center")
    table.add_column("Visible", justify="center")

    for dataset in snapshot.datasets:
        table.add_row(
            str(dataset.get("id", "-")),
            str(dataset.get("label", "-")),
            str(dataset.get("payload_endpoint", "-")),
            _fmt_bool(dataset.get("payload_requires_api_key", True)),
            _fmt_bool(dataset.get("help_visible", False)),
        )

    console.print(table)


def show_example_universes(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_red]Example Universes[/bold bright_red]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("ID", style="bright_cyan", no_wrap=True)
    table.add_column("Label", style="bold bright_white")
    table.add_column("Symbols", overflow="fold")

    for universe in snapshot.example_universes:
        table.add_row(
            str(universe.get("id", "-")),
            str(universe.get("label", "-")),
            _fmt_list(universe.get("symbols")),
        )

    console.print(table)


def show_available_symbols(snapshot: QJDataSnapshot) -> None:
    table = Table(
        box=box.DOUBLE_EDGE,
        title="[bold bright_cyan]Available Symbols[/bold bright_cyan]",
        header_style="bold bright_white",
        row_styles=["none", "dim"],
    )
    table.add_column("Symbol", style="bold bright_green", no_wrap=True)
    table.add_column("Universe", overflow="fold")

    for item in snapshot.available_symbols:
        table.add_row(
            str(item.get("symbol", "-")),
            _fmt_list(item.get("universe_labels")),
        )

    console.print(table)


def show_source_detail(source: dict[str, object]) -> None:
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Source ID", str(source.get("id", "-"))),
                    ("Label", str(source.get("label", "-"))),
                    ("Visible in help", _fmt_bool(source.get("public_help_visible", False))),
                    (
                        "Prepare requires API key",
                        _fmt_bool(source.get("requires_api_key_for_prepare", True)),
                    ),
                    ("Notes", str(source.get("notes", "-"))),
                ]
            ),
            title=_make_title_text("Source Detail"),
            border_style="bright_green",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )


def show_granularity_detail(granularity: dict[str, object]) -> None:
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Granularity ID", str(granularity.get("id", "-"))),
                    ("Category", str(granularity.get("category", "-"))),
                    ("Label", str(granularity.get("label", "-"))),
                    ("Supported sources", _fmt_list(granularity.get("supported_sources"))),
                ]
            ),
            title=_make_title_text("Granularity Detail"),
            border_style="bright_magenta",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )


def show_dataset_detail(dataset: dict[str, object]) -> None:
    endpoint = str(dataset.get("payload_endpoint", "-"))
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Dataset ID", str(dataset.get("id", "-"))),
                    ("Label", str(dataset.get("label", "-"))),
                    ("Payload endpoint", endpoint),
                    (
                        "Payload requires API key",
                        _fmt_bool(dataset.get("payload_requires_api_key", True)),
                    ),
                    ("Visible in help", _fmt_bool(dataset.get("help_visible", False))),
                ]
            ),
            title=_make_title_text("Dataset Detail"),
            border_style="bright_yellow",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )
    console.print(
        _make_list_panel(
            "Dataset Notes",
            [
                f"Payload route: {endpoint}" if endpoint != "-" else "No payload route exposed.",
                (
                    "Public metadata does not expose dataset row counts, per-symbol coverage, "
                    "or freshness by asset."
                ),
            ],
            border_style="bright_blue",
        )
    )


def show_universe_detail(universe: dict[str, object]) -> None:
    symbols = [str(symbol) for symbol in universe.get("symbols", [])]
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Universe ID", str(universe.get("id", "-"))),
                    ("Label", str(universe.get("label", "-"))),
                    ("Symbol count", str(len(symbols))),
                ]
            ),
            title=_make_title_text("Universe Detail"),
            border_style="bright_red",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )
    console.print(_make_list_panel("Universe Symbols", symbols, border_style="bright_cyan"))


def show_symbol_detail(symbol_item: dict[str, object], snapshot: QJDataSnapshot) -> None:
    public_granularities = [str(item.get("id", "-")) for item in snapshot.granularities]
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Symbol", str(symbol_item.get("symbol", "-"))),
                    ("Found in universes", str(symbol_item.get("universe_count", 0))),
                    (
                        "Asset-specific granularity coverage",
                        _fmt_status(str(symbol_item.get("granularity_status", "Not exposed"))),
                    ),
                    (
                        "Asset-specific period coverage",
                        _fmt_status(str(symbol_item.get("period_status", "Not exposed"))),
                    ),
                    (
                        "Asset-specific start/end dates",
                        _fmt_status(str(symbol_item.get("date_range_status", "Not exposed"))),
                    ),
                ]
            ),
            title=_make_title_text("Symbol Detail"),
            border_style="bright_cyan",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )
    console.print(
        Columns(
            [
                _make_list_panel(
                    "Seen In Example Universes",
                    [str(label) for label in symbol_item.get("universe_labels", [])],
                    border_style="bright_green",
                ),
                _make_list_panel(
                    "Public Platform Granularities",
                    public_granularities,
                    border_style="bright_magenta",
                ),
            ],
            equal=True,
            expand=True,
        )
    )
    console.print(
        Panel(
            (
                "The public metadata endpoints currently expose platform-level granularities and "
                "example universes, but they do not expose per-symbol period coverage, per-symbol "
                "granularity coverage, or per-symbol date ranges."
            ),
            title=_make_title_text("Availability Note"),
            border_style="bright_yellow",
            box=box.ROUNDED,
            padding=(1, 1),
        )
    )


def show_asset_class_detail(asset_class: str, snapshot: QJDataSnapshot) -> None:
    console.print(
        Panel(
            _make_info_table(
                [
                    ("Asset class", asset_class),
                    ("Direct asset membership", _fmt_status("Not exposed by public metadata")),
                    (
                        "Asset-level periods and dates",
                        _fmt_status("Not exposed by public metadata"),
                    ),
                    (
                        "Asset-level granularity coverage",
                        _fmt_status("Not exposed by public metadata"),
                    ),
                ]
            ),
            title=_make_title_text("Asset Class Detail"),
            border_style="bright_blue",
            box=box.HEAVY,
            padding=(1, 1),
        )
    )
    console.print(
        _make_list_panel(
            "Public Platform Granularities",
            [str(item.get("id", "-")) for item in snapshot.granularities],
            border_style="bright_magenta",
        )
    )


def show_error(message: str) -> None:
    console.print(
        Panel(
            Text(message, style="bold white"),
            title=_make_title_text("Error"),
            border_style="bright_red",
            box=box.DOUBLE,
            padding=(1, 1),
        )
    )
