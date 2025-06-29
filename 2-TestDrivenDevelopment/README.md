# Test-Driven Development

Test-Driven Development is a software development process where you:

- Write a failing test first for a small piece of functionality.
- Write the minimal code to make the test pass.
- Refactor the code (clean it up), making sure tests still pass.

This cycle is often called Red–Green–Refactor:

- Red: Write a test that fails.
- Green: Write code to pass the test.
- Refactor: Improve the code without changing behavior.

Can add `@pytest.mark.smoke` decorators to select or deselect tests in pytest.
