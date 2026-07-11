"""
qj_data — interactive public metadata browser
---------------------------------------------

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, cast

import questionary
from questionary import Choice

from backtester.cli.qj_data_api import DEFAULT_API_BASE_URL, fetch_qj_data_snapshot
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
        prog="qj-data",
        description="Browse QuantJourney backtester metadata in the terminal.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"Public QuantJourney API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        metavar="SECONDS",
        help="HTTP timeout in seconds (default: 20).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        with console.status(
            "[bold bright_cyan]Loading QuantJourney public metadata...[/bold bright_cyan]",
            spinner="dots12",
            spinner_style="bright_magenta",
        ):
            snapshot = fetch_qj_data_snapshot(
                base_url=args.base_url,
                timeout=args.timeout,
            )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        show_error(f"Failed to load metadata: {exc}")
        return 1

    while True:
        try:
            clear_screen()
            show_home_banner(snapshot)

            section = cast(
                str,
                _select(
                    "Select a section",
                    [
                        Choice("View All", "view_all", shortcut_key="0"),
                        Choice("Overview", "overview"),
                        Choice("Symbols in example universes", "symbols"),
                        Choice("Sources", "sources"),
                        Choice("Granularities", "granularities"),
                        Choice("Asset classes", "asset_classes"),
                        Choice("Datasets", "datasets"),
                        Choice("Example universes", "universes"),
                        Choice("Exit", "exit"),
                    ],
                ),
            )

            clear_screen()

            if section == "exit":
                return 0
            if section == "view_all":
                clear_screen()
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


if __name__ == "__main__":
    raise SystemExit(main())
