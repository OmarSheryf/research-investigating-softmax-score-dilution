from pathlib import Path
from playwright.sync_api import sync_playwright

workspace=Path(__file__).resolve().parents[1]
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True,timeout=15000)
    page=browser.new_page(viewport={'width':736,'height':1400})
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((workspace/'outputs/causal-run/explorer.html').as_uri(),wait_until='domcontentloaded',timeout=15000)
    frame=page.frame_locator('iframe')
    frame.locator('#cc-bars .cc-row').first.wait_for()
    for width in [320,736,1024]:
        page.set_viewport_size({'width':width,'height':1600})
        overflow=frame.locator('#causal-checkpoints').evaluate('(root)=>({width:root.clientWidth,scroll:document.documentElement.scrollWidth,viewport:innerWidth})')
        assert overflow['scroll']<=overflow['viewport']+1,overflow
        page.screenshot(path=str(workspace/f'work/causal-explorer-{width}.png'),full_page=True)
    frame.locator('[data-key="seed"]').select_option('33')
    frame.locator('[data-key="metric"]').select_option('single')
    frame.locator('[data-key="step"]').select_option('400')
    assert 'checkpoint 400' in frame.locator('#cc-status').inner_text()
    assert frame.locator('#cc-bars .cc-row').count()==2
    assert not errors,errors
    print('Browser passed: initial render, controls, no JS errors, no page overflow at 320/736/1024px.')
    browser.close()
