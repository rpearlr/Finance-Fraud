"""
Clean up residual old topbar fragments that the patch left behind.
The old topbar had a .logo div that wasn't captured by the regex properly.
"""
import glob, re

FILES = sorted(glob.glob('frontend/*.html'))

# The residual from the old topbar that the regex didn't catch
# Pattern: leftover after the new topbar was injected
RESIDUAL_PATTERN = re.compile(
    r'\s*FinRisk AI\s*\n\s*</div>\s*\n\s*<div class="context">[^<]*</div>\s*\n\s*<div class="status">[^<]*●[^<]*</div>\s*\n\s*</div>',
    re.DOTALL
)

for filepath in FILES:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    before = content
    content = RESIDUAL_PATTERN.sub('', content)
    
    if content != before:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Cleaned residual from: {filepath}")
    else:
        print(f"No residual in: {filepath}")

print("Done.")
