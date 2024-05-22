SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: viewcoverage coverage setup help install uninstall diagrams buildr buildd test clean debug release sample updatebadge doc

f_release = build_Release
f_debug = build_Debug
f_diagrams = diagrams
app_targets = BayesNet
test_targets = TestBayesNet
clang-uml = clang-uml
plantuml = plantuml
lcov = lcov
genhtml = genhtml
dot = dot
n_procs = -j 16

define ClearTests
	@for t in $(test_targets); do \
		if [ -f $(f_debug)/tests/$$t ]; then \
			echo ">>> Cleaning $$t..." ; \
			rm -f $(f_debug)/tests/$$t ; \
		fi ; \
	done
	@nfiles="$(find . -name "*.gcda" -print0)" ; \
	if test "${nfiles}" != "" ; then \
		find . -name "*.gcda" -print0 | xargs -0 rm 2>/dev/null ;\
	fi ; 
endef


setup: ## Install dependencies for tests and coverage
	@if [ "$(shell uname)" = "Darwin" ]; then \
		brew install gcovr; \
		brew install lcov; \
	fi
	@if [ "$(shell uname)" = "Linux" ]; then \
		pip install gcovr; \
		sudo dnf install lcov;\
	fi
	@echo "* You should install plantuml & graphviz for the diagrams"

diagrams: ## Create an UML class diagram & depnendency of the project (diagrams/BayesNet.png)
	@which $(plantuml) || (echo ">>> Please install plantuml"; exit 1)
	@which $(dot) || (echo ">>> Please install graphviz"; exit 1)
	@which $(clang-uml) || (echo ">>> Please install clang-uml"; exit 1)
	@export PLANTUML_LIMIT_SIZE=16384
	@echo ">>> Creating UML class diagram of the project...";
	@$(clang-uml) -p 
	@cd $(f_diagrams); \
	$(plantuml) -tsvg BayesNet.puml
	@echo ">>> Creating dependency graph diagram of the project...";
	$(MAKE) debug
	cd $(f_debug) && cmake .. --graphviz=dependency.dot 
	@$(dot) -Tsvg $(f_debug)/dependency.dot.BayesNet -o $(f_diagrams)/dependency.svg

buildd: ## Build the debug targets
	cmake --build $(f_debug) -t $(app_targets) $(n_procs)

buildr: ## Build the release targets
	cmake --build $(f_release) -t $(app_targets) $(n_procs)

clean: ## Clean the tests info
	@echo ">>> Cleaning Debug BayesNet tests...";
	$(call ClearTests)
	@echo ">>> Done";

uninstall: ## Uninstall library
	@echo ">>> Uninstalling BayesNet...";
	xargs rm < $(f_release)/install_manifest.txt
	@echo ">>> Done";

prefix = "/usr/local"
install: ## Install library
	@echo ">>> Installing BayesNet...";
	@cmake --install $(f_release) --prefix $(prefix)
	@echo ">>> Done";

debug: ## Build a debug version of the project
	@echo ">>> Building Debug BayesNet...";
	@if [ -d ./$(f_debug) ]; then rm -rf ./$(f_debug); fi
	@mkdir $(f_debug); 
	@cmake -S . -B $(f_debug) -D CMAKE_BUILD_TYPE=Debug -D ENABLE_TESTING=ON -D CODE_COVERAGE=ON
	@echo ">>> Done";

release: ## Build a Release version of the project
	@echo ">>> Building Release BayesNet...";
	@if [ -d ./$(f_release) ]; then rm -rf ./$(f_release); fi
	@mkdir $(f_release); 
	@cmake -S . -B $(f_release) -D CMAKE_BUILD_TYPE=Release
	@echo ">>> Done";

fname = "tests/data/iris.arff"
sample: ## Build sample
	@echo ">>> Building Sample...";
	@if [ -d ./sample/build ]; then rm -rf ./sample/build; fi
	@cd sample && cmake -B build -S . && cmake --build build -t bayesnet_sample
	sample/build/bayesnet_sample $(fname)
	@echo ">>> Done";	

opt = ""
test: ## Run tests (opt="-s") to verbose output the tests, (opt="-c='Test Maximum Spanning Tree'") to run only that section
	@echo ">>> Running BayesNet tests...";
	@$(MAKE) clean
	@cmake --build $(f_debug) -t $(test_targets) $(n_procs)
	@for t in $(test_targets); do \
		echo ">>> Running $$t...";\
		if [ -f $(f_debug)/tests/$$t ]; then \
			cd $(f_debug)/tests ; \
			./$$t $(opt) ; \
			cd ../.. ; \
		fi ; \
	done
	@echo ">>> Done";

coverage: ## Run tests and generate coverage report (build/index.html)
	@echo ">>> Building tests with coverage..."
	@which $(lcov) || (echo ">>> Please install lcov"; exit 1)
	@if [ ! -f $(f_debug)/tests/coverage.info ] ; then $(MAKE) test ; fi
	@echo ">>> Building report..."
	@cd $(f_debug)/tests; \
	$(lcov) --directory CMakeFiles --capture --demangle-cpp --ignore-errors source,source --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info '/usr/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'lib/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'libtorch/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'tests/*' --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info 'bayesnet/utils/loguru.*' --ignore-errors unused --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --remove coverage.info '/opt/miniconda/*' --ignore-errors unused --output-file coverage.info >/dev/null 2>&1; \
	$(lcov) --summary coverage.info
	@$(MAKE) updatebadge
	@echo ">>> Done";	

viewcoverage: ## View the html coverage report
	@which $(genhtml) || (echo ">>> Please install lcov (genhtml not found)"; exit 1)
	@$(genhtml) $(f_debug)/tests/coverage.info --demangle-cpp --output-directory html --title "BayesNet Coverage Report" -s -k -f --legend >/dev/null 2>&1;
	@xdg-open html/index.html || open html/index.html 2>/dev/null
	@echo ">>> Done";

updatebadge: ## Update the coverage badge in README.md
	@which python || (echo ">>> Please install python"; exit 1)
	@if [ ! -f $(f_debug)/tests/coverage.info ]; then \
		echo ">>> No coverage.info file found. Run make coverage first!"; \
		exit 1; \
	fi
	@echo ">>> Updating coverage badge..."
	@env python update_coverage.py $(f_debug)/tests
	@echo ">>> Done";

doc: ## Generate documentation
	@echo ">>> Generating documentation..."
	@cmake --build $(f_release) -t doxygen
	@echo ">>> Done";

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
