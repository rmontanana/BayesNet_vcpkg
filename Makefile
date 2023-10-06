SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: coverage setup help build test clean debug release

setup: ## Install dependencies for tests and coverage
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install gcovr; \
		brew install lcov; \
	fi
	@if [ "$(shell uname)" = "Linux" ]; then \
		pip install gcovr; \
	fi

dest ?= ${HOME}/bin
install: ## Copy binary files to bin folder
	@echo "Destination folder: $(dest)"
	make buildr
	@echo ">>> Copying files to $(dest)"
	@cp build_release/src/Platform/b_main $(dest)
	@cp build_release/src/Platform/b_list $(dest)
	@cp build_release/src/Platform/b_manage $(dest)
	@cp build_release/src/Platform/b_best $(dest)

dependency: ## Create a dependency graph diagram of the project (build/dependency.png)
	cd build && cmake .. --graphviz=dependency.dot && dot -Tpng dependency.dot -o dependency.png

buildd: ## Build the debug targets
	cmake --build build_debug -t b_main -t BayesNetSample -t b_manage -t b_list -t b_best -j 32

buildr: ## Build the release targets
	cmake --build build_release -t b_main -t BayesNetSample -t b_manage -t b_list -t b_best -j 32

clean: ## Clean the debug info
	@echo ">>> Cleaning Debug BayesNet...";
	$(call ClearTests)
	@echo ">>> Done";

clang-uml: ## Create uml class and sequence diagrams
	clang-uml -p --add-compile-flag -I /usr/lib/gcc/x86_64-redhat-linux/8/include/

debug: ## Build a debug version of the project
	@echo ">>> Building Debug BayesNet...";
	@if [ -d ./build_debug ]; then rm -rf ./build_debug; fi
	@mkdir build_debug; 
	@cmake -S . -B build_Debug -D CMAKE_BUILD_TYPE=Debug -D ENABLE_TESTING=ON -D CODE_COVERAGE=ON;
	@echo ">>> Done";

release: ## Build a Release version of the project
	@echo ">>> Building Release BayesNet...";
	@if [ -d ./build_release ]; then rm -rf ./build_release; fi
	@mkdir build_release; 
	@cmake -S . -B build_release -D CMAKE_BUILD_TYPE=Release; 
	@echo ">>> Done";	

opt = ""
test: ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running BayesNet & Platform tests...";
	$(MAKE) clean
	@cmake --build build_debug --target unit_tests_bayesnet --target unit_tests_platform ; 
	@if [ -f build_debug/tests/unit_tests_bayesnet ]; then cd build_debug/tests ; ./unit_tests_bayesnet $(opt) ; fi ; 
	@if [ -f build_debug/tests/unit_tests_platform ]; then cd build_debug/tests ; ./unit_tests_platform $(opt) ; fi ; 
	@echo ">>> Done";

opt = ""
testp: ## Run platform tests (opt="-s") to verbose output the tests, (opt="-c='Stratified Fold Test'") to run only that section
	@echo ">>> Running Platform tests...";
	$(MAKE) clean
	@cmake --build build_debug --target unit_tests_platform ; 
	@if [ -f build_debug/tests/unit_tests_platform ]; then cd build_debug/tests ; ./unit_tests_platform $(opt) ; fi ; 
	@echo ">>> Done";

opt = ""
testb: ## Run BayesNet tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running BayesNet tests...";
	$(MAKE) clean
	@cmake --build build_debug --target unit_tests_bayesnet ; 
	@if [ -f build_debug/tests/unit_tests_bayesnet ]; then cd build_debug/tests ; ./unit_tests_bayesnet $(opt) ; fi ; 
	@echo ">>> Done";

coverage: ## Run tests and generate coverage report (build/index.html)
	@echo ">>> Building tests with coverage...";
	$(MAKE) test
	@cd build_debug ; \
	gcovr --config ../gcovr.cfg
	@echo ">>> Done";	

define ClearTests =
	$(eval nfiles=$(find . -name "*.gcda" -print))
	@if [ -f build_debug/tests/unit_tests_bayesnet ]; then rm -f build_debug/tests/unit_tests_bayesnet ; fi ; 
	@if [ -f build_debug/tests/unit_tests_platform ]; then rm -f build_debug/tests/unit_tests_platform ; fi ; 
	@if test "${nfiles}" != "" ; then \
		find . -name "*.gcda" -print0 | xargs -0 rm 2>/dev/null ;\
	fi ; 
endef

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
