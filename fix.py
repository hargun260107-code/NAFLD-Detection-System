with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

fixed = content.replace('"CRP":                  crp,', '"CRP ":                 crp,')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(fixed)

print('Done!')