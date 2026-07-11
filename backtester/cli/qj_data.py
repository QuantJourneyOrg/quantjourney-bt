"""
qj_data — public backtester data-catalog CLI
--------------------------------------------

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any, cast

import questionary
from questionary import Choice

from backtester.cli.qj_data_api import (
    DEFAULT_API_BASE_URL,
    QJDataSnapshot,
    fetch_qj_data_snapshot,
)
from backtester.cli.qj_data_views import (
    clear_screen,
    console,
    show_asset_class_detail,
    show_asset_classes,
    show_dataset_detail,
    show_datasets,
    show_error,
    show_example_symbols,
    show_example_universes,
    show_granularities,
    show_home_banner,
    show_overview,
    show_source_detail,
    show_sources,
    show_symbol_detail,
    show_universe_detail,
    show_view_all,
)
from backtester.version import __version__

SECTIONS = (
    "overview",
    "sources",
    "granularities",
    "datasets",
    "asset-classes",
    "universes",
    "example-symbols",
    "all",
)


def _select(message: str, choices: list[Choice]) -> Any:
    answer = questionary.select(
        message,
        choices=choices,
        qmark="",
        use_shortcuts=True,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _after_view() -> str:
    return cast(
        str,
        _select(
            "Choose next step",
            [
                Choice("Back to main menu", "main"),
                Choice("Exit", "exit"),
            ],
        ),
    )


def _item_choice_label(item: dict[str, Any]) -> str:
    return f"{item.get('id', '-')}  |  {item.get('label', '-')}"


def _prompt_symbol(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    symbol_map = {str(item.get("symbol", "")).upper(): item for item in items}

    while True:
        answer = questionary.text(
            "Type a symbol from the table to open details, or press Enter to go back"
        ).ask()
        if answer is None:
            raise KeyboardInterrupt

        normalized = answer.strip().upper()
        if not normalized:
            return None
        if normalized in symbol_map:
            return symbol_map[normalized]

        show_error(f"Symbol '{answer.strip()}' was not found in the table above.")


def _browse_items(
    *,
    prompt: str,
    items: list[dict[str, Any]],
    make_choice_label: Callable[[dict[str, Any]], str],
    render_item: Callable[[dict[str, Any]], None],
) -> str:
    while True:
        choices = [Choice(make_choice_label(item), item) for item in items]
        choices.append(Choice("Back", "__back__"))

        selected = _select(prompt, choices)
        if selected == "__back__":
            return "main"

        clear_screen()
        render_item(cast(dict[str, Any], selected))
        if _after_view() == "exit":
            return "exit"
        clear_screen()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qj-bt",
        description="QuantJourney Backtester command-line tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    data_parser = commands.add_parser(
        "data",
        help="Browse the public backtester data catalog.",
        description=(
            "Browse public sources, granularities, datasets, example universes, "
            "and symbols referenced by those examples."
        ),
    )
    data_parser.add_argument(
        "section",
        nargs="?",
        choices=SECTIONS,
        help="Catalog section (interactive in a TTY when omitted).",
    )
    data_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit stable machine-readable JSON instead of Rich tables.",
    )
    data_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Force the keyboard-driven catalog browser.",
    )
    data_parser.add_argument(
        "--base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"Public QuantJourney API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    data_parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        metavar="SECONDS",
        help="HTTP timeout in seconds (default: 20).",
    )
    return parser


def _catalog_metadata(snapshot: QJDataSnapshot) -> dict[str, Any]:
    return {
        "catalog_revision": snapshot.catalog_doc.get("catalog_revision"),
        "schema_version": snapshot.catalog_doc.get("schema_version"),
        "snapshot_date": snapshot.catalog_doc.get("snapshot_date")
        or snapshot.help_doc.get("snapshot_date"),
        "source": snapshot.source_label,
    }


def _section_data(snapshot: QJDataSnapshot, section: str) -> Any:
    if section == "overview":
        return {
            "asset_classes": len(snapshot.asset_classes),
            "sources": len(snapshot.sources),
            "granularities": len(snapshot.granularities),
            "datasets": len(snapshot.datasets),
            "example_universes": len(snapshot.example_universes),
            "example_symbols": len(snapshot.example_symbols),
            "strategies": snapshot.catalog_doc.get("strategy_summary", {}),
        }
    if section == "sources":
        return snapshot.sources
    if section == "granularities":
        return snapshot.granularities
    if section == "datasets":
        return snapshot.datasets
    if section == "asset-classes":
        return snapshot.asset_classes
    if section == "universes":
        return snapshot.example_universes
    if section == "example-symbols":
        return snapshot.example_symbols
    if section == "all":
        return {
            "asset_classes": snapshot.asset_classes,
            "sources": snapshot.sources,
            "granularities": snapshot.granularities,
            "datasets": snapshot.datasets,
            "example_universes": snapshot.example_universes,
            "example_symbols": snapshot.example_symbols,
            "strategy_summary": snapshot.catalog_doc.get("strategy_summary", {}),
        }
    raise ValueError(f"Unknown qj-bt data section: {section}")


def _show_section(snapshot: QJDataSnapshot, section: str) -> None:
    renderers: dict[str, Callable[[QJDataSnapshot], None]] = {
        "overview": show_overview,
        "sources": show_sources,
        "granularities": show_granularities,
        "datasets": show_datasets,
        "asset-classes": show_asset_classes,
        "universes": show_example_universes,
        "example-symbols": show_example_symbols,
        "all": show_view_all,
    }
    renderers[section](snapshot)


def _show_json(snapshot: QJDataSnapshot, section: str) -> None:
    payload = {
        **_catalog_metadata(snapshot),
        "section": section,
        "data": _section_data(snapshot, section),
    }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run_interactive(snapshot: QJDataSnapshot) -> int:
    while True:
        try:
            clear_screen()
            show_home_banner(snapshot)

            section = cast(
                str,
                _select(
                    "Select a section",
                    [
                        Choice("Overview", "overview"),
                        Choice("Sources", "sources"),
                        Choice("Granularities", "granularities"),
                        Choice("Datasets", "datasets"),
                        Choice("Asset classes", "asset_classes"),
                        Choice("Example universes", "universes"),
                        Choice("Symbols in example universes", "symbols"),
                        Choice("View all", "view_all"),
                        Choice("Exit", "exit"),
                    ],
                ),
            )

            clear_screen()
            if section == "exit":
                return 0
            if section == "view_all":
                show_view_all(snapshot)
                if _after_view() == "exit":
                    return 0
            elif section == "overview":
                show_overview(snapshot)
                if _after_view() == "exit":
                    return 0
            elif section == "symbols":
                while True:
                    show_example_symbols(snapshot)
                    selected_symbol = _prompt_symbol(snapshot.example_symbols)
                    if selected_symbol is None:
                        break

                    clear_screen()
                    show_symbol_detail(selected_symbol, snapshot)
                    if _after_view() == "exit":
                        return 0
                    clear_screen()
            elif section == "sources":
                show_sources(snapshot)
                if (
                    _browse_items(
                        prompt="Select a source",
                        items=snapshot.sources,
                        make_choice_label=_item_choice_label,
                        render_item=show_source_detail,
                    )
                    == "exit"
                ):
                    return 0
            elif section == "granularities":
                show_granularities(snapshot)
                if _after_view() == "exit":
                    return 0
            elif section == "asset_classes":
                show_asset_classes(snapshot)
                asset_choices = [Choice(str(item), item) for item in snapshot.asset_classes]
                asset_choices.append(Choice("Back", "__back__"))
                selected = _select("Asset class details", asset_choices)
                if selected != "__back__":
                    clear_screen()
                    show_asset_class_detail(str(selected), snapshot)
                    if _after_view() == "exit":
                        return 0
            elif section == "datasets":
                show_datasets(snapshot)
                if (
                    _browse_items(
                        prompt="Select a dataset",
                        items=snapshot.datasets,
                        make_choice_label=_item_choice_label,
                        render_item=show_dataset_detail,
                    )
                    == "exit"
                ):
                    return 0
            elif section == "universes":
                show_example_universes(snapshot)
                if (
                    _browse_items(
                        prompt="Select an example universe",
                        items=snapshot.example_universes,
                        make_choice_label=_item_choice_label,
                        render_item=show_universe_detail,
                    )
                    == "exit"
                ):
                    return 0
        except KeyboardInterrupt:
            return 130


def _is_interactive_request(args: argparse.Namespace) -> bool:
    if args.interactive:
        return True
    return args.section is None and not args.as_json and sys.stdin.isatty() and sys.stdout.isatty()


def _fetch_snapshot(args: argparse.Namespace, *, interactive: bool) -> QJDataSnapshot:
    if not interactive:
        return fetch_qj_data_snapshot(base_url=args.base_url, timeout=args.timeout)
    with console.status(
        "[bold bright_cyan]Loading QuantJourney public metadata...[/bold bright_cyan]",
        spinner="dots12",
        spinner_style="bright_magenta",
    ):
        return fetch_qj_data_snapshot(base_url=args.base_url, timeout=args.timeout)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.interactive and (args.section is not None or args.as_json):
        parser.error("--interactive cannot be combined with a section or --json")

    interactive = _is_interactive_request(args)
    section = args.section or "overview"
    try:
        snapshot = _fetch_snapshot(args, interactive=interactive)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if args.as_json:
            print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        else:
            show_error(f"Failed to load metadata: {exc}")
        return 1

    if interactive:
        return _run_interactive(snapshot)
    if args.as_json:
        _show_json(snapshot, section)
    else:
        _show_section(snapshot, section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
