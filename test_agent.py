import json
import sys
import os
import time
from dotenv import load_dotenv
from collections import defaultdict

# Environment değişkenlerini yükle
load_dotenv()

from langchain_core.messages import HumanMessage
from agents.agent import build_graph, analyze_question_type

def load_test_questions(file_path="data/metadata.jsonl"):
    """Load test questions from metadata.jsonl file"""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            questions.append({
                "task_id": data["task_id"],
                "question": data["Question"],
                "expected_answer": data["Final answer"],
                "level": data["Level"],
                "tools": data.get("Annotator Metadata", {}).get("Tools", ""),
                "steps": data.get("Annotator Metadata", {}).get("Number of steps", "")
            })
    return questions

def categorize_questions(questions):
    """Categorize questions by type and level"""
    categories = defaultdict(list)
    levels = defaultdict(list)
    
    for q in questions:
        # Categorize by question type
        q_type = analyze_question_type(q["question"])
        categories[q_type].append(q)
        
        # Categorize by difficulty level
        levels[f"Level_{q['level']}"].append(q)
    
    return categories, levels

def test_single_question(graph, question_data, verbose=True):
    """Test a single question"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing Task ID: {question_data['task_id']}")
        print(f"Level: {question_data['level']}")
        print(f"Question: {question_data['question'][:150]}...")
        print(f"Expected Answer: {question_data['expected_answer']}")
    
    start_time = time.time()
    
    messages = [HumanMessage(content=question_data['question'])]
    result = graph.invoke({"messages": messages})
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    answer = result["messages"][-1].content
    
    # Check if answer is correct (exact match or contains expected answer)
    is_correct = (
        question_data['expected_answer'].lower() == answer.lower() or
        question_data['expected_answer'].lower() in answer.lower()
    )
    
    if verbose:
        print(f"Agent Answer: {answer}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Match: {'✅' if is_correct else '❌'}")
    
    return {
        "task_id": question_data['task_id'],
        "answer": answer,
        "expected": question_data['expected_answer'],
        "is_correct": is_correct,
        "execution_time": execution_time
    }

def test_by_category(graph, questions, category_name, category_questions):
    """Test all questions in a specific category"""
    print(f"\n{'='*80}")
    print(f"Testing Category: {category_name}")
    print(f"Number of questions: {len(category_questions)}")
    print('='*80)
    
    results = []
    correct_count = 0
    
    for i, q in enumerate(category_questions):
        print(f"\nQuestion {i+1}/{len(category_questions)}")
        result = test_single_question(graph, q, verbose=True)
        results.append(result)
        if result['is_correct']:
            correct_count += 1
    
    # Summary for category
    accuracy = (correct_count / len(category_questions)) * 100 if category_questions else 0
    avg_time = sum(r['execution_time'] for r in results) / len(results) if results else 0
    
    print(f"\n{'-'*40}")
    print(f"Category: {category_name} - Summary")
    print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(category_questions)})")
    print(f"Average execution time: {avg_time:.2f} seconds")
    print(f"{'-'*40}\n")
    
    return results

def generate_report(all_results):
    """Generate a comprehensive test report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Overall statistics
    total_questions = sum(len(results) for results in all_results.values())
    total_correct = sum(sum(1 for r in results if r['is_correct']) for results in all_results.values())
    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    
    # Category breakdown
    print("\n\nCATEGORY BREAKDOWN:")
    print("-"*50)
    
    for category, results in all_results.items():
        if results:
            correct = sum(1 for r in results if r['is_correct'])
            accuracy = (correct / len(results)) * 100
            avg_time = sum(r['execution_time'] for r in results) / len(results)
            
            print(f"\n{category}:")
            print(f"  Questions: {len(results)}")
            print(f"  Correct: {correct}")
            print(f"  Accuracy: {accuracy:.1f}%")
            print(f"  Avg Time: {avg_time:.2f}s")
    
    # Failed questions details
    print("\n\nFAILED QUESTIONS DETAILS:")
    print("-"*50)
    
    for category, results in all_results.items():
        failed = [r for r in results if not r['is_correct']]
        if failed:
            print(f"\n{category}:")
            for r in failed:
                print(f"  Task ID: {r['task_id']}")
                print(f"  Expected: {r['expected']}")
                print(f"  Got: {r['answer']}")
                print()
    
    # Save report to file
    with open('test_report.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\n\nDetailed report saved to: test_report.json")

def main():
    # Build the graph
    print("Building agent graph...")
    graph = build_graph()
    
    # Load test questions
    print("Loading test questions...")
    test_questions = load_test_questions()
    
    # Categorize questions
    categories, levels = categorize_questions(test_questions)
    
    # Test options
    print("\n\nTest Options:")
    print("1. Test all questions")
    print("2. Test by category")
    print("3. Test by difficulty level")
    print("4. Test specific question IDs")
    print("5. Test first N questions")
    
    choice = input("\nSelect option (1-5): ")
    
    all_results = {}
    
    if choice == "1":
        # Test all questions by category
        for category, questions in categories.items():
            results = test_by_category(graph, test_questions, category, questions)
            all_results[category] = results
    
    elif choice == "2":
        # Test specific category
        print("\nAvailable categories:")
        for i, cat in enumerate(categories.keys()):
            print(f"{i+1}. {cat} ({len(categories[cat])} questions)")
        
        cat_choice = int(input("\nSelect category: ")) - 1
        selected_category = list(categories.keys())[cat_choice]
        
        results = test_by_category(graph, test_questions, selected_category, categories[selected_category])
        all_results[selected_category] = results
    
    elif choice == "3":
        # Test by difficulty level
        print("\nAvailable levels:")
        for i, level in enumerate(levels.keys()):
            print(f"{i+1}. {level} ({len(levels[level])} questions)")
        
        level_choice = int(input("\nSelect level: ")) - 1
        selected_level = list(levels.keys())[level_choice]
        
        results = test_by_category(graph, test_questions, selected_level, levels[selected_level])
        all_results[selected_level] = results
    
    elif choice == "4":
        # Test specific question IDs
        task_ids = input("\nEnter task IDs (comma-separated): ").split(',')
        selected_questions = [q for q in test_questions if q['task_id'].strip() in [id.strip() for id in task_ids]]
        
        results = test_by_category(graph, test_questions, "Selected Questions", selected_questions)
        all_results["Selected Questions"] = results
    
    elif choice == "5":
        # Test first N questions
        n = int(input("\nEnter number of questions to test: "))
        selected_questions = test_questions[:n]
        
        results = test_by_category(graph, test_questions, f"First {n} Questions", selected_questions)
        all_results[f"First {n} Questions"] = results
    
    # Generate comprehensive report
    generate_report(all_results)

if __name__ == "__main__":
    main()