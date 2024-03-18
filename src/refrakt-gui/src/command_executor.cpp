#include <command_executor.h>

void command_executor::execute(command_t&& command) {
	command.command();

	if (can_redo()) {
		command_stack.erase(current_command, command_stack.end());
	}

	command_stack.emplace_back(std::move(command));
	current_command = command_stack.end();
}

void command_executor::undo() {
	if (!can_undo())
		return;

	--current_command;
	if (!current_command->description.empty()) SPDLOG_INFO("Undoing: {}", current_command->description);
	current_command->undoer();
}

void command_executor::redo() {
	if (!can_redo())
		return;

	current_command->command();
	if (!current_command->description.empty()) SPDLOG_INFO("Redoing: {}", current_command->description);
	++current_command;
}
