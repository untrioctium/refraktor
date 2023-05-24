#pragma once

class command_executor {
public:

	command_executor() : current_command{ command_stack.end() } {}

	struct command_t {
		thunk_t command;
		thunk_t undoer;
		std::string description;
	};

	void execute(command_t&& command) {
		command.command();

		if (can_redo()) {
			command_stack.erase(current_command, command_stack.end());
		}

		command_stack.emplace_back(std::move(command));
		current_command = command_stack.end();
	}

	void execute(thunk_t&& command, thunk_t&& undoer, std::string_view description = {}) {
		execute({ std::move(command), std::move(undoer), std::string{description} });
	}

	bool can_undo() const {
		return current_command != command_stack.begin();
	}

	bool can_redo() const {
		return current_command != command_stack.end();
	}

	void undo() {
		if (!can_undo())
			return;

		--current_command;
		if(!current_command->description.empty()) SPDLOG_INFO("Undoing: {}", current_command->description);
		current_command->undoer();
	}

	void redo() {
		if (!can_redo())
			return;

		current_command->command();
		if (!current_command->description.empty()) SPDLOG_INFO("Redoing: {}", current_command->description);
		++current_command;
	}

	void clear() {
		command_stack.clear();
		current_command = command_stack.end();
	}

private:

	std::list<command_t> command_stack{};
	decltype(command_stack)::iterator current_command;

};