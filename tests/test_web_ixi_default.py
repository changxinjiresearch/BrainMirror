from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_ixi_population_autoload_is_source_verified_and_truthful():
    js=(ROOT/'site'/'app.js').read_text(encoding='utf8')
    html=(ROOT/'site'/'index.html').read_text(encoding='utf8')
    assert 'loadIXIPopulation' in js
    assert "await loadIXIPopulation();" in js
    assert "crypto.subtle.digest('SHA-256'" in js
    assert '36b25fbf1f6749b309cc93adc811ee2137fa83fdab89cf72ba87886e542cda21' in js
    assert "json.provenance.subjects_included!==467" in js
    assert "json.provenance.status!=='RESEARCH_CANDIDATE_UNREVIEWED'" in js
    assert 'loadIXIBtn' in html
    assert 'CC BY-SA 3.0' in html
    assert '尚未完成' in html or '尚未批准' in html or '未获独立' in js
    assert 'NOT LINKED' in html
