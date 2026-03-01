# /commit - Smart Git Commit

Automatically stage, commit, and optionally push changes with a well-formatted conventional commit message.

## Usage

```bash
/commit              # Commit all changes
/commit --push       # Commit and push
/commit --amend      # Amend last commit (use with caution)
```

## Workflow

1. **Check Status**: Run `git status` and `git diff --stat` to understand changes
2. **Stage Files**: Add relevant files (avoid .env, credentials, large binaries)
3. **Generate Message**: Create conventional commit message based on actual changes
4. **Commit**: Execute commit with proper formatting
5. **Push** (optional): Push to remote if `--push` specified and remote exists

## Commit Message Format

```
<type>(<scope>): <subject>

[body - optional, for complex changes]

Co-authored-by: Claude <noreply@anthropic.com>
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(query): add spatial relation support` |
| `fix` | Bug fix | `fix(slam): handle empty point cloud` |
| `refactor` | Code restructure | `refactor(llm): use environment variables` |
| `docs` | Documentation | `docs: update CLAUDE.md with architecture` |
| `test` | Test changes | `test(query): add e2e query tests` |
| `chore` | Maintenance | `chore: update dependencies` |
| `perf` | Performance | `perf(mapping): optimize similarity computation` |

### Subject Rules

- Use imperative mood: "add feature" not "added feature"
- Keep under 72 characters
- Describe the change, not the files modified
- No trailing period

### Scope (Optional)

Common scopes for this project:
- `query` - Query scene components
- `slam` - SLAM/mapping pipeline
- `llava` - VLM integration
- `scenegraph` - Scene graph building
- `bash` - Shell scripts

## Examples

**Good commits:**
```
feat(query): implement nested spatial query parsing

Add support for complex queries like "the pillow on the sofa near the door".
Parser now handles recursive spatial constraints.

Co-authored-by: Claude <noreply@anthropic.com>
```

```
fix(bash): use environment variables instead of hardcoded paths

Co-authored-by: Claude <noreply@anthropic.com>
```

```
refactor(llm): consolidate model configuration

- Move API keys to environment variables
- Add MODEL_CONFIGS mapping for all supported models
- Remove hardcoded endpoints from scripts

Co-authored-by: Claude <noreply@anthropic.com>
```

**Bad commits:**
```
[BAD] update files                    # Too vague
[BAD] fix: modified query_parser.py   # Describes file, not change
[BAD] WIP                             # Not descriptive
[BAD] feat: add new feature :rocket:  # Contains emoji
```

## Safety Rules

1. **Never commit secrets**: Check for API keys, tokens, passwords
2. **Review large diffs**: For 500+ lines, summarize sections in body
3. **Preserve existing co-authors**: Don't remove other contributors
4. **Check remote before push**: Only push if remote exists
5. **Don't amend pushed commits**: Use new commit instead

## Pre-commit Checks

Before committing, verify:
- [ ] No `console.log` statements (unless intentional)
- [ ] No hardcoded paths or credentials
- [ ] Tests pass (if applicable)
- [ ] Linting passes (if configured)

## Error Handling

- **No changes**: Skip commit, inform user
- **Git identity not set**: Prompt user to configure
- **Push fails**: Report error, don't auto-retry
- **Hook fails**: Fix issue, create new commit (don't amend)
