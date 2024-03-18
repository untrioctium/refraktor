#pragma once

class command_executor {
public:

	command_executor() : current_command{ command_stack.end() } {}

	struct command_t {
		thunk_t command;
		thunk_t undoer;
		std::string description;
	};

	void execute(command_t&& command);

	void execute(thunk_t&& command, thunk_t&& undoer, std::string_view description = {}) {
		execute({ std::move(command), std::move(undoer), std::string{description} });
	}

	bool can_undo() const {
		return current_command != command_stack.begin();
	}

	bool can_redo() const {
		return current_command != command_stack.end();
	}

	void undo();

	void redo();

	void clear() {
		command_stack.clear();
		current_command = command_stack.end();
	}

private:

	std::list<command_t> command_stack{};
	decltype(command_stack)::iterator current_command;

};