# PR Review Skill: C++ Code Review

## Description
Review C++ code changes in the OpenVINO GenAI project for code quality, performance, and best practices.

## Trigger
When reviewing PRs that modify files in `src/cpp/` or `tests/cpp/`

## Review Checklist

### Code Quality
- [ ] Code follows C++17 standards used in the project
- [ ] Proper use of smart pointers (`std::shared_ptr`, `std::unique_ptr`) instead of raw pointers
- [ ] RAII principles are followed for resource management
- [ ] No memory leaks or dangling pointers
- [ ] Proper const-correctness (const references, const methods)
- [ ] Move semantics used where appropriate

### Header Files
- [ ] Include guards or `#pragma once` present
- [ ] Minimal includes in headers (forward declarations preferred)
- [ ] No circular dependencies introduced
- [ ] Public API headers in `src/cpp/include/openvino/` are backward compatible

### Shared Modules Impact
- [ ] Check if changes affect shared utilities (Logger, Tokenizer, LoRA, SynchronizedQueue)
- [ ] Verify backward compatibility if modifying shared code
- [ ] Ensure all dependent components are tested

### Performance Considerations
- [ ] No unnecessary copies (use references, move semantics)
- [ ] Avoid allocations in hot paths
- [ ] Thread safety for shared resources (check `SynchronizedQueue` usage)
- [ ] Proper use of OpenVINO inference request pooling

### Error Handling
- [ ] Exceptions are used appropriately
- [ ] Error messages are descriptive and include context
- [ ] Resources are properly cleaned up on error paths

### Testing
- [ ] Unit tests added/updated in `tests/cpp/`
- [ ] Test coverage for edge cases
- [ ] Tests follow existing patterns (see `tests/cpp/helper.hpp`)

## Common Issues to Flag

1. **Breaking API changes** in public headers without deprecation
2. **Thread safety issues** in continuous batching or streaming code
3. **Missing null checks** for optional parameters
4. **Hardcoded values** that should be configurable
5. **Missing logging** for error conditions
