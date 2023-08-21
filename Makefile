SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage setup help build test

setup: ## Install dependencies for tests and coverage
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install gcovr; \
		brew install lcov; \
	fi
	@if [ "$(shell uname)" = "Linux" ]; then \
		pip install gcovr; \
	fi

copy: ## Copy binary files to selected folder
	@echo "Destination folder: $(dest)"
	@if test "$(dest)" == "" ; then \
		echo "Destination folder is empty"; \
		exit 1; \
	fi
	make build
	@echo ">>> Copying files to $(dest)"
	@cp build/src/Platform/main $(dest)
	@cp build/src/Platform/list $(dest)
	@cp build/src/Platform/manage $(dest)
	@echo ">>> Done"

dependency: ## Create a dependency graph diagram of the project (build/dependency.png)
	cd build && cmake .. --graphviz=dependency.dot && dot -Tpng dependency.dot -o dependency.png

build: ## Build the main and BayesNetSample
	cmake --build build -t main -t BayesNetSample -t manage -t list -j 32

clean: ## Clean the debug info
	@echo ">>> Cleaning Debug BayesNet ...";
	find . -name "*.gcda" -print0 | xargs -0 rm
	@echo ">>> Done";

debug: ## Build a debug version of the project
	@echo ">>> Building Debug BayesNet ...";
	@if [ -d ./build ]; then rm -rf ./build; fi
	@mkdir build; 
	cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug -D ENABLE_TESTING=ON -D CODE_COVERAGE=ON; \
	cmake --build build -j 32;
	@echo ">>> Done";

release: ## Build a Release version of the project
	@echo ">>> Building Release BayesNet ...";
	@if [ -d ./build ]; then rm -rf ./build; fi
	@mkdir build; 
	cmake -S . -B build -D CMAKE_BUILD_TYPE=Release; \
	cmake --build build -t main -t BayesNetSample -t manage -t list -j 32;
	@echo ">>> Done";	

test: ## Run tests
	@echo "* Running tests...";
	find . -name "*.gcda" -print0 | xargs -0 rm
	@cd build; \
	cmake --build . --target unit_tests ;
	@cd build/tests; \
	./unit_tests;

coverage: ## Run tests and generate coverage report (build/index.html)
	@echo "*Building tests...";
	find . -name "*.gcda" -print0 | xargs -0 rm
	@cd build; \
	cmake --build . --target unit_tests ;
	@cd build/tests; \
	./unit_tests;
	gcovr ;

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done
