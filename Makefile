# Single entry point for humans and Codex. Replace the TODO lines once the stack is chosen.
.PHONY: install dev check

install:
	@echo "TODO: install deps, e.g. (cd apps/web && npm ci) && (cd apps/api && pip install -r requirements.txt)"

dev:
	@echo "TODO: run web + api, e.g. npx concurrently 'make -C apps/web dev' 'make -C apps/api dev'"

check:
	@echo "TODO: lint + tests, e.g. (cd apps/web && npm run lint) && (cd apps/api && pytest -q)"
