"""Интеграция: локальные xlsx -> analysis pipeline с моделью БР -> секция бюллетеня."""

from __future__ import annotations

import json

import pytest

from ai4epi.analysis.analysis_pipeline import AnalysisOutputConfig, AnalysisPipelineSettings, run_analysis_pipeline
from ai4epi.analysis.mechanistic_wave import MechanisticWaveConfig
from ai4epi.analysis.preprocessing import FeatureEngineeringConfig
from ai4epi.core.sections import default_section_registry
from ai4epi.data.local_excel import load_local_influenza
from ai4epi.output.figures import build_publication_figure_assets
from ai4epi.quality.runtime_checks import RuntimeChecker

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def analysis(data_dir, tmp_path_factory):
    bundle = load_local_influenza("spb", data_dir=data_dir, end_date="2019-01-21")
    out = tmp_path_factory.mktemp("analysis")
    result = run_analysis_pipeline(
        influenza_weekly=bundle.weekly,
        age_group_frame=bundle.cases,
        settings=AnalysisPipelineSettings(require_weather=False, run_mechanistic_wave=True, raise_on_error=True),
        feature_config=FeatureEngineeringConfig(temp_cols=()),
        mechanistic_wave_config=MechanisticWaveConfig(method="annealing"),
        output=AnalysisOutputConfig(output_dir=out),
    )
    return result, out


def test_context_contains_mechanistic_block(analysis):
    result, _ = analysis
    assert result.status in {"ok", "partial"}
    ctx = result.context
    block = ctx.get_block("mechanistic_wave")
    assert block["season_label"] == "2018–2019"
    assert [h["target_date"] for h in block["horizons"]] == [h.target_date.isoformat() for h in ctx.forecast.horizons]
    assert block["comparison_with_ml"]["agreement_code"] in {"agree", "disagree"}


def test_section_evidence_and_runtime_numeric_anchor(analysis):
    result, _ = analysis
    ctx = result.context
    registry = default_section_registry()
    section = registry.get("mechanistic_wave")
    assert section.is_available(ctx)
    payload = section.build_user_payload(ctx)
    evidence = payload["mechanistic_wave_context"]
    assert evidence["peak"]["value"] == ctx.get_block("mechanistic_wave")["peak"]["value"]

    peak = evidence["peak"]["value"]
    week = evidence["peak"]["iso_week"]
    good = {
        "mechanistic_wave": {
            "phase_text": f"Максимум ожидается на {week}-й неделе на уровне {peak} на 10 тыс.",
            "comparison_text": "Прогнозы расходятся.",
        }
    }
    bad = {"mechanistic_wave": {"phase_text": "Максимум ожидается вскоре.", "comparison_text": "Прогнозы расходятся."}}
    checker = RuntimeChecker()
    def anchor_issues(sections):
        report = checker.check_bulletin(context=ctx, bulletin=sections, registry=registry, section_ids=["mechanistic_wave"])
        return [i for i in report.issues if i.section_id == "mechanistic_wave" and "Барояна" in i.message]
    assert not anchor_issues(good)
    assert anchor_issues(bad)


def test_figure_generated(analysis):
    result, out = analysis
    assets = build_publication_figure_assets(
        context=result.context, analysis_artifacts=result.artifacts, output_dir=out / "figures"
    )
    assert "mechanistic_wave_plot" in assets
    assert assets["mechanistic_wave_plot"]["metadata"]["section_id"] == "mechanistic_wave"


def test_section_skipped_without_block(analysis):
    result, _ = analysis
    data = result.context.to_public_dict()
    data.pop("mechanistic_wave")
    from ai4epi.core.context import GlobalContext

    ctx = GlobalContext.model_validate(json.loads(json.dumps(data)))
    assert not default_section_registry().get("mechanistic_wave").is_available(ctx)
