#include <iostream>
#include <random>
#include<utility>
#include <unordered_map>
#include <iostream>
#include <random>
#include <cmath>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

// Read Data from CSV
vector<vector<float>> ReadDataFromCSV(const string& filename, int rows, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;

    }

    vector<vector<float>> data(rows, vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float value;
            char dummy;

            file >> value;
            data[i][j] = value;

            // So the dummy won't eat digits
            if (j < (cols - 1))
                file >> dummy;
        }
    }

    file.close();
    return data;
}

//Expression Tree (chromosome) 
//Binary Tree
class BinaryNode
{
protected:
    float data;
    BinaryNode* left;
    BinaryNode* right;
    BinaryNode* parent;
public:
    BinaryNode(float val = 0.0f, BinaryNode* l = nullptr, BinaryNode* r = nullptr)
        :data(val), left(l), right(r) {}
    BinaryNode(char val, BinaryNode* l = nullptr, BinaryNode* r = nullptr)
        :data(val), left(l), right(r) {}
    void setData(float val) { data = val; }
    void setLeft(BinaryNode* l) { left = l; }
    void setRight(BinaryNode* r) { right = r; }
    float getData() { return data; }
    int getData2() { return data; }
    BinaryNode* getLeft() { return left; }
    BinaryNode* getRight() { return right; }
    bool isLeaf() { return left == NULL && right == NULL; }
    char getOperator() {
        if (isLeaf()) {
            return '\0';
        }
        return static_cast<char>(data);
    }
    BinaryNode* getParent() const { return parent; }

    bool hasLeft() const { return left != nullptr; }
    bool hasRight() const { return right != nullptr; }

};

class BinaryTree
{
    BinaryNode* root;
public:
    BinaryTree() : root(NULL) {}

    void rm(BinaryNode* node) {
        if (node == nullptr)    return;
        rm(node->getLeft());
        rm(node->getRight());
        delete node;
    }

    BinaryTree* Copy() {
        BinaryTree* n = new BinaryTree();

        n->setRoot(cp(root));

        return n;
    }

    BinaryNode* cp(BinaryNode* node) {
        if (node == nullptr)
            return nullptr;
        BinaryNode* n = new BinaryNode(node->getData(), nullptr, nullptr);
        n->setLeft(cp(node->getLeft()));
        n->setRight(cp(node->getRight()));

        return n;
    }

    void setRoot(BinaryNode* node) { root = node; }
    BinaryNode* getRoot() { return root; }
    bool isEmpty() { return root == NULL; }
    //BinaryTree recursion calculation
    void inorder(BinaryNode* node) {
        if (node == nullptr)
            return;

        inorder(node->getLeft());
        if (node->isLeaf()) {
            std::cout << node->getData() << " ";
        }
        else {
            char op = static_cast<char>(node->getData());
            std::cout << op << " ";
        }
        inorder(node->getRight());
    }
    void inorder() {
        inorder(root);
    }
    int getHeight() { return isEmpty() ? 0 : getHeight(root); }
    int getHeight(BinaryNode* node)
    {
        if (node == NULL) {
            return 0;
        }
        int hLeft = getHeight(node->getLeft());
        int hRight = getHeight(node->getRight());
        return (hLeft > hRight) ? hLeft + 1 : hRight + 1;
    }
    int getPrecedence(char op) {
        switch (op) {
        case 's':
        case 'c':
        case 't':
        case 'h':
        case 'o':
        case 'a':
        case 'l':
        case 'e':
            return 4;  // Highest precedence for functions
        case '*':
        case '/':
            return 3;
        case '+':
        case '-':
            return 2;
        default:
            return 1;  // Lowest precedence for leaf nodes
        }
    }

    void post(BinaryNode* node)
    {
        if (node == nullptr)
            return;

        post(node->getLeft());
        post(node->getRight());

        if (node->isLeaf())
        {
            cout << node->getData() << " ";
        }
        else
        {
            cout << static_cast<char>(node->getData()) << " ";
        }
    }

    void inorder2(BinaryNode* node, int parentPrecedence = 0) {
        if (node == nullptr)
            return;

        int currentPrecedence = getPrecedence(node->getData());

        if (currentPrecedence > parentPrecedence && !node->isLeaf())
            std::cout << "(";

        if (node->isLeaf()) {
            float value = node->getData();
            string leafChar;

            if (value == 0.0f)
                leafChar = "A";
            else if (value == 1.0f)
                leafChar = "B";
            else if (value == 2.0f)
                leafChar = "C";
            else if (value == 3.0f)
                leafChar = "D";
            else if (value == 4.0f)
                leafChar = "E";
            else if (value == 5.0f)
                leafChar = "F";
            else
                leafChar = to_string(value);  // handle other leaf values as needed

            std::cout << leafChar;
        }
        else {
            char op = static_cast<char>(node->getData());
            switch (op) {
            case '+':
                std::cout << "(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case '-':
                std::cout << "(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '-';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case '*':
                std::cout << "(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '*';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case '/':
                std::cout << "(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '/';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 's':
                //std::cout << "np.sin(";
                std::cout << "sin(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'c':
                //std::cout << "np.cos(";
                std::cout << "cos(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 't':
                //std::cout << "np.tan(";
                std::cout << "tan(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'h':
                //std::cout << "np.sinh(";
                std::cout << "sinh(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'o':
                //std::cout << "np.cosh(";
                std::cout << "cosh(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'a':
                //std::cout << "np.tanh(";
                std::cout << "tanh(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'l':
                //std::cout << "np.log(";
                std::cout << "log(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            case 'e':
                //std::cout << "np.exp(";
                std::cout << "exp(";
                inorder2(node->getLeft(), currentPrecedence);
                std::cout << '+';
                inorder2(node->getRight(), currentPrecedence);
                std::cout << ")";
                break;
            }
        }

        if (currentPrecedence > parentPrecedence && !node->isLeaf()) {
            std::cout << ")";
        }
    }
    string inorder_key(BinaryNode* node) {
        if (node == nullptr)
            return "";

        string result;
        result += inorder_key(node->getLeft());
        if (node->isLeaf()) {
            result += std::to_string(node->getData()) + " ";
        }
        else {
            char op = static_cast<char>(node->getData());
            result += op;
            result += " ";
        }
        result += inorder_key(node->getRight());

        return result;
    }

    string inorder_key() {
        return inorder_key(root);
    }
    //void preorder() {}
    //void postorder() {}
    //void levelorder() {}

    //BinaryTree additional calculation
    // int getCount() {}
    //int getHeight() {}
    //int getLeafCount() {}
    //Expression Tree member function
    //Expression Tree that Function  that has node as a root due to its recursion call

    //Evaluate Cost of each expression tree
    float evaluate() { return evaluate(root); }

    float evaluate(BinaryNode* node)
    {
        if (node == nullptr)
            return std::numeric_limits<float>::quiet_NaN();

        if (node->isLeaf())
            return node->getData();
        else
        {
            float op1 = evaluate(node->getLeft());
            float op2 = evaluate(node->getRight());

            switch (node->getData2()) {
            case '+':
                return op1 + op2;
                break;
            case '-':
                return op1 - op2;
                break;
            case '*':
                return op1 * op2;
                break;
            case '/':
                if (op2 != 0) {
                    float result = op1 / op2;
                    return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();
                }
                else {
                    return std::numeric_limits<float>::quiet_NaN();  // Return NaN for division by zero
                }
                break;
            case 's':
            {
                float sum = op1 + op2;
                float result = std::sin(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 'c':
            {
                float sum = op1 + op2;
                float result = std::cos(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 't':
            {
                float sum = op1 + op2;
                float result = std::tan(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 'h':
            {
                float sum = op1 + op2;
                float result = std::sinh(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 'o':
            {
                float sum = op1 + op2;
                float result = std::cosh(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 'a':
            {
                float sum = op1 + op2;
                float result = std::tanh(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            case 'l':
            {
                float sum = op1 + op2;
                float result = std::log(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
            }
            case 'e':
            {
                float sum = op1 + op2;
                float result = std::exp(sum);
                return std::isfinite(result) ? result : std::numeric_limits<float>::quiet_NaN();  // Return NaN for non-finite result
                break;
            }
            }
        }
        return pow(2, 31);
    }
};

BinaryTree generateExpressionTree2() {
    std::random_device rd_;
    std::mt19937 gen_(rd_());
    std::vector<char> operators = { '+', '-', '*','/','s','c','t','h','o','a','l','e' };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> numOperandsDist(2, 6);  // Number of operands (2 to 6)
    int numOperands = numOperandsDist(gen);
    int numOperators = numOperands - 1;

    std::vector<BinaryNode*> operands(numOperands);
    for (int i = 0; i < numOperands; ++i) {
        operands[i] = new BinaryNode(0.0f, nullptr, nullptr);
    }
    if (numOperands != 1)
    {
        std::uniform_int_distribution<int> operatorDist(0, operators.size() - 1);
        for (int i = 0; i < numOperators; ++i) {
            char selectedOperator = operators[operatorDist(gen)];
            BinaryNode* operatorNode = new BinaryNode(selectedOperator, nullptr, nullptr);
            operatorNode->setLeft(operands[i]);
            operatorNode->setRight(operands[i + 1]);
            operands[i + 1] = operatorNode;
        }
    }
    BinaryTree tree;
    tree.setRoot(operands.back());
    return tree;
}

//Customize Dataset
int CountOperands(BinaryNode* node) {
    if (node == nullptr) {
        return 0;
    }

    if (node->isLeaf()) {
        return 1;
    }
    int leftOperands = CountOperands(node->getLeft());
    int rightOperands = CountOperands(node->getRight());

    return leftOperands + rightOperands;
}

vector<float>At(6);
vector<vector<float>> FinData(BinaryTree tree, vector<vector<float>> data)
{
    BinaryNode* root = tree.getRoot();
    int Operandnum = CountOperands(root);
    std::vector<float> attributes = { 0, 1, 2, 3, 4, 5 };
    std::vector<float> selectedAttributes(attributes.size());
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < attributes.size(); i++)
    {
        std::uniform_int_distribution<int> dist(0, attributes.size() - 1);
        int index = dist(gen);
        float selectedAttribute = attributes[index];
        selectedAttributes[i] = selectedAttribute;
    }
    // Clear the original attributes vector
    attributes.clear();

    // Repeat the selected attributes six times
    for (int i = 0; i < selectedAttributes.size(); i++) {
        attributes.insert(attributes.end(), selectedAttributes.begin(), selectedAttributes.end());
    }

    vector <int> consAt;
    vector <float> consVal(Operandnum);
    for (int k = 0; k < Operandnum; k++)
    {
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        float prob = 0.16f;
        if (probDist(gen) <= prob)
        {
            consAt.push_back(k);
            std::uniform_real_distribution<float> constNum(-10.00f, 10.00f);
            float constnum = std::roundf(constNum(gen) * 100) / 100;
            attributes[k] = 6;
            consVal[k] = constnum;
        }
        else
        {
            // consVal[k] = 100;
        }
    }
    attributes.resize(Operandnum);


    vector<vector<float>> FinData(200, vector<float>(attributes.size()));
    // Resize FinData to have Operandnum rows and attributes.size() columns
    for (int i = 0; i < 200; ++i) {
        for (int j = 0; j < attributes.size(); ++j) {
            if (attributes[j] == 6)
            {
                FinData[i][j] = consVal[j];
            }
            else
                FinData[i][j] = data[i][attributes[j]];
        }
    }

    for (int l = 0; l < consAt.size(); l++)
    {
        attributes[consAt[l]] = consVal[consAt[l]];
    }

    At.clear();

    At = attributes;


    return FinData;
}


void AttributeValues(BinaryNode* node, const vector<float>& attr, int& dataIndex) {
    if (node == nullptr) {
        return;
    }

    if (node->isLeaf()) {
        node->setData(attr[dataIndex]);
        dataIndex++;
    }
    else {
        AttributeValues(node->getLeft(), attr, dataIndex);
        AttributeValues(node->getRight(), attr, dataIndex);
    }
}
void revertAttributeValues(BinaryTree& tree, const vector<float>& data) {
    int dataIndex = 0;
    AttributeValues(tree.getRoot(), data, dataIndex);
}

//Insert Operands and Calculate
void revertNodeValues(BinaryNode* node, const vector<float>& data, int& dataIndex) {
    if (node == nullptr) {
        return;
    }

    if (node->isLeaf()) {
        node->setData(data[dataIndex]);

        dataIndex++;
    }
    else {
        revertNodeValues(node->getLeft(), data, dataIndex);
        revertNodeValues(node->getRight(), data, dataIndex);
    }
}

void revertExpressionTree(BinaryTree& tree, const vector<float>& data) {
    int dataIndex = 0;
    revertNodeValues(tree.getRoot(), data, dataIndex);
}

std::random_device rd;
std::mt19937 gen(rd());

BinaryNode* getRandomEdgeAtLevel(BinaryNode* node, int level) {
    if (level == 1) {
        return node;
    }
    else if (level > 1) {
        BinaryNode* leftEdge = getRandomEdgeAtLevel(node->getLeft(), level - 1);
        BinaryNode* rightEdge = getRandomEdgeAtLevel(node->getRight(), level - 1);

        // Randomly select one of the edges at the current level
        std::uniform_int_distribution<int> dist(0, 1);
        int choice = dist(gen);

        return (choice == 0) ? leftEdge : rightEdge;
    }

    return nullptr;
}

BinaryNode* selectRandomEdge(BinaryNode* node) {
    if (node == nullptr || node->isLeaf()) {
        return nullptr;
    }

    BinaryTree binaryTree; // Create an instance of BinaryTree
    int treeHeight = binaryTree.getHeight();  // Call getHeight() on the BinaryTree instance

    if (treeHeight < 2) {
        return nullptr; // Handle the case when the tree height is less than 2
    }

    std::uniform_int_distribution<int> dist(1, treeHeight - 1);  // Select level between 1 and treeHeight - 1
    int level = dist(gen); // Use the existing random number generator

    return getRandomEdgeAtLevel(node, level);
}

bool areBothLeftChildren(BinaryNode* parent, BinaryNode* child) {
    return (parent != nullptr && child != nullptr && parent->getLeft() == child);
}

bool areBothRightChildren(BinaryNode* parent, BinaryNode* child) {
    return (parent != nullptr && child != nullptr && parent->getRight() == child);
}

void swapSubtrees(BinaryNode* node, BinaryNode* node_, BinaryNode* edge, BinaryNode* edge_) {
    if (node == nullptr || node_ == nullptr || edge == nullptr || edge_ == nullptr) {
        return;
    }

    // Check if the selected edges are both left or right children of their respective parents
    if (areBothLeftChildren(node, edge) && areBothLeftChildren(node_, edge_)) {
        BinaryNode* tmp = node->getLeft();
        node->setLeft(node_->getLeft());
        node_->setLeft(tmp);
    }
    else if (areBothRightChildren(node, edge) && areBothRightChildren(node_, edge_)) {
        BinaryNode* tmp = node->getRight();
        node->setRight(node_->getRight());
        node_->setRight(tmp);
    }
    else {
        // The selected edges are not compatible for swapping, so don't perform the swap
        return;
    }

    // Recursively swap subtrees
    swapSubtrees(node->getLeft(), node_->getLeft(), edge, edge_);
    swapSubtrees(node->getRight(), node_->getRight(), edge, edge_);
}

BinaryTree crossover(BinaryTree p1, BinaryTree p2) {
    BinaryTree parent1 = p1;
    BinaryTree parent2 = p2;

    // Randomly select edges at the same level for crossover
    BinaryNode* edge1 = selectRandomEdge(parent1.getRoot());
    BinaryNode* edge2 = selectRandomEdge(parent2.getRoot());

    // Swap subtrees if the edges are compatible
    swapSubtrees(parent1.getRoot(), parent2.getRoot(), edge1, edge2);

    return parent1;
}
//Mutation: Change the operator node value randomly so that it contains an operator other than the original one
//Available Change: Random Node point
BinaryNode* getRandomNode(BinaryTree& tree)
{
    BinaryNode* node = tree.getRoot();
    std::random_device rd;
    std::mt19937 gen(rd());

    while (true)
    {
        if (!node->isLeaf() && node->getOperator() != '\0') // Check if the node is a non-leaf node and contains an operator
        {
            return node;
        }
        else if (node->hasLeft() || node->hasRight())
        {
            std::uniform_int_distribution<int> choiceDist(0, 1); // Randomly choose among left child and right child
            int choice = choiceDist(gen);
            if (choice == 0 && node->hasLeft())
            {
                node = node->getLeft();
            }
            else if (choice == 1 && node->hasRight())
            {
                node = node->getRight();
            }
        }
        else
        {
            // If both left and right child are absent, it is a leaf node
            // No non-leaf node with an operator is available, so return nullptr or handle the case appropriately based on your requirements.
            return nullptr;
        }
    }

    return node;
}
void mutateExpressionTree(BinaryTree& tree)
{
    BinaryNode* node = getRandomNode(tree);
    std::random_device rd;
    std::mt19937 gen(rd());
    //choices of mutation probability
    std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
    float mutprob = 0.01f; // 1%
    if (probDist(gen) <= mutprob)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        //choices of Operators (Default choice:'+' choice1: '-',choice2: '*, choice3: '/')
        std::uniform_int_distribution<int> operatorDist(0, 10);
        if (!node->isLeaf())
        {
            int selectedOpIndex = operatorDist(gen);
            char selectedOp = '+';
            switch (selectedOpIndex)
            {
            case 0:
                selectedOp = '-';
                break;
            case 1:
                selectedOp = '*';
                break;
            case 2:
                selectedOp = '/';
                break;
            case 3:
                selectedOp = 's';
                break;
            case 4:
                selectedOp = 'c';
                break;
            case 5:
                selectedOp = 't';
                break;
            case 6:
                selectedOp = 'h';
                break;
            case 7:
                selectedOp = 'o';
                break;
            case 8:
                selectedOp = 'a';
                break;
            case 9:
                selectedOp = 'l';
                break;
            case10:
                selectedOp = 'e';
                break;

            }
            node->setData(selectedOp);
        }
    }
}

//Local Search
//modifyNode: Change the Operator Randomly
void modifyNode(BinaryNode* node) {
    std::random_device rd;
    std::mt19937 gen(rd());
    if (!node->isLeaf()) {
        std::uniform_int_distribution<int> operatorDist(0, 10);
        int selectedOpIndex = operatorDist(gen);
        char selectedOp = '+';
        switch (selectedOpIndex) {
        case 0:
            selectedOp = '-';
            break;
        case 1:
            selectedOp = '*';
            break;
        case 2:
            selectedOp = '/';
            break;
        case 3:
            selectedOp = 's';
            break;
        case 4:
            selectedOp = 'c';
            break;
        case 5:
            selectedOp = 't';
            break;
        case 6:
            selectedOp = 'h';
            break;
        case 7:
            selectedOp = 'o';
            break;
        case 8:
            selectedOp = 'a';
            break;
        case 9:
            selectedOp = 'l';
            break;
        case 10:
            selectedOp = 'e';
            break;
        }
        node->setData(selectedOp);
    }
}
// Count the number of operators in the tree
int countOperators(BinaryNode* node) {
    if (node == nullptr) {
        return 0;
    }

    int count = 0;

    if (!node->isLeaf()) {
        count++;
    }

    count += countOperators(node->getLeft());
    count += countOperators(node->getRight());

    return count;
}
//Local Optimization
void localSearch(BinaryTree& tree) {
    bool improvement = true;

    while (improvement) {
        float currentFitness = tree.evaluate(); // Evaluate the fitness of the current tree

        improvement = false;

        // Perform local search on each node in the tree
        for (int i = 0; i < countOperators(tree.getRoot()); i++) {
            BinaryTree modifiedTree = tree; // Create a copy of the original tree

            BinaryNode* node = getRandomNode(modifiedTree); // Get a random non-leaf node

            // Apply a local modification to the node
            modifyNode(node);

            float modifiedFitness = modifiedTree.evaluate(); // Evaluate the fitness of the modified tree

            // If the modified tree has better fitness, replace the original tree
            if (modifiedFitness < currentFitness) {
                tree = modifiedTree;
                currentFitness = modifiedFitness;
                improvement = true;
            }
        }
    }
}

//Calculate Cost (MSE)
float Cost(vector<vector<float>> data, vector<float> resCal)
{
    vector<float> cost(200);
    float cos = 0.0f;
    bool hasNaN = false; // Flag to track if NaN is encountered
    for (int j = 0; j < 200; j++) {
        vector<float> D = data[j];
        if (std::isnan(resCal[j]) || std::isnan(D[6])) {
            hasNaN = true; // Set the flag if resCal[j] or D[6] is NaN
            break; // No need to calculate further, exit the loop
        }

        float squaredDifference = pow((resCal[j] - D[6]), 2);
        //float squaredDifference = resCal[j];
        if (std::isnan(squaredDifference) || std::isinf(squaredDifference)) {
            hasNaN = true; // Set the flag if squaredDifference is NaN or inf
            break; // No need to calculate further, exit the loop
        }

        cost[j] = squaredDifference;
        cos += cost[j];
    }

    if (hasNaN) {
        return pow(2, 31); // If NaN is encountered, return 0.0
    }
    else {
        cos = cos / 2.0f;
        return cos; // Otherwise, return the calculated cost
    }
}


/*
// Memoization을 위한 unordered_map 선언
unordered_map<string, float> cal; // Memoization map
unordered_map<string, float> memo; // Memoization map

// Recursive function to calculate Cost
float calculateCost(vector<vector<float>>& data, vector<float>& resCal, int index, string key, string memo_key) {

    if (index >= 200) {
        // 모든 데이터에 대한 적합도 값 계산 완료
        return cal[key]; // memo에 저장된 값 반환
    }

    if (memo.find(memo_key) != memo.end()) {
        // memo에 저장된 값이 있으면 반환
        return memo[memo_key];
    }

    // 현재 데이터에 대한 적합도 계산
    vector<float> D = data[index];

    if (isnan(resCal[index]) || isnan(D[6])) {
        cal[key] = pow(2, 31); // Set memoized value as 2^31(큰 수)
    }
    else {
        float squaredDifference = pow((resCal[index] - D[6]), 2);

        if (isnan(squaredDifference) || isinf(squaredDifference)) {
            // 다음 데이터로 재귀 호출
            cal[key] = pow(2, 31); // Set memoized value as 2^31(큰 수) if squaredDifference is NaN or inf
        }
        else {
            // 다음 데이터로 재귀 호출
            // 현재 데이터의 적합도 값과 다음 데이터의 적합도 값을 더해서 memo에 저장
            // 각 키마다 특정 해(tree 조합)에 해당하는 0~199까지의 적합도 합 저장
            float nextCost = calculateCost(data, resCal, index + 1, key, memo_key);
            //cout << "index: " << index << " , key: " << key << " , cost: " << cal[key] << endl;
            cal[key] = squaredDifference + nextCost;

        }
    }
    return cal[key];
}

float Cost(vector<vector<float>>& data, vector<float>& resCal, BinaryTree tree) {
    cal.clear(); // cal map 초기화
    string key = " ";
    string memo_key = tree.inorder_key(tree.getRoot());
    float cost = calculateCost(data, resCal, 0, key, memo_key);
    cost = cost / 2.0f;
    memo[memo_key] = cost;

    // If any NaN value encountered, return 0.0
    if (isnan(cost)) {
        return pow(2, 31);
    }

    return cost;
}
*/


pair<float, BinaryTree> best_population;
static bool compare(const pair<float, BinaryTree>& a, const pair<float, BinaryTree>& b) {
    return a.first < b.first;
}
pair<float, BinaryTree> geneticAlgorithm(vector<vector<float>>& data, int population_size, int generation_size)
{
    // Create initial population
    vector<BinaryTree> population(population_size);
    vector<float> fitness(population_size);
    for (int i = 0; i < population_size; ++i) {
        population[i] = generateExpressionTree2();
        localSearch(population[i]);
    }

    // Calculate fitness of each chromosome in the initial population(Lower the Better)
    for (int i = 0; i < population_size; i++) {
        At.clear();
        vector<float> resCal(200);
        vector<vector<float>> finData = FinData(population[i], data);
        for (int j = 0; j < 200; j++) {
            vector<float> Data = finData[j];
            revertExpressionTree(population[i], Data);
            resCal[j] = population[i].evaluate();
        }

        revertAttributeValues(population[i], At);
        fitness[i] = Cost(data, resCal);

    }

    // Repeat Generation
    for (int generation = 0; generation < generation_size; generation++) {
        //Selection- Elitism
        int parent1_idx = 0;
        int parent2_idx = 0;

        vector<float>revertfitness = fitness;
        parent1_idx = distance(revertfitness.begin(), min_element(revertfitness.begin(), revertfitness.end()));
        revertfitness.erase(revertfitness.begin() + parent1_idx);
        parent2_idx = distance(revertfitness.begin(), min_element(revertfitness.begin(), revertfitness.end()));

        BinaryTree* parent1 = population[parent1_idx].Copy();
        BinaryTree* parent2 = population[parent2_idx].Copy();
        //Local Search Each parent
        localSearch(*parent1);
        localSearch(*parent2);

        // Crossover
        BinaryTree child = crossover(*parent1, *parent2);

        // Mutation
        mutateExpressionTree(child);
        //Local Search
        localSearch(child);


        // Evaluate fitness of child chromosome
        vector<float> childResCal(200);

        At.clear();
        vector<vector<float>> child_finData = FinData(child, data); // Calculate only using specific attributes

        for (int i = 0; i < 200; i++) {
            vector<float> Data = child_finData[i];
            revertExpressionTree(child, Data);
            childResCal[i] = child.evaluate();
        }
        revertAttributeValues(child, At);
        //float child_fitness = Cost(data, childResCal, child);
        float child_fitness = Cost(data, childResCal);

        // Replace
        // Replace Chromosomej with the worst fitness with the child chromosome
        int worstIndex = distance(fitness.begin(), max_element(fitness.begin(), fitness.end()));

        if (child_fitness < fitness[worstIndex]) {
            population[worstIndex] = child; // Replace the worst fitness chromosome with the child expression tree
            fitness[worstIndex] = child_fitness;
        }

        //Replacement2: Replace the chromosomes with high fitness by generating new trees again 
        //Pair the fitness and the population index    
        vector <pair<float, int>> sortfitness(fitness.size());
        for (int k = 0; k < fitness.size(); k++)
        {
            sortfitness[k].first = fitness[k];
            sortfitness[k].second = k;
        }
        //Sort chromosomes in descending order to find the high fitness chromosomes.
        sort(sortfitness.begin(), sortfitness.end(), greater<pair<float, int>>());
        //10% of the high fitness chromosomes are replaced with newly generated trees
        for (int j = 0; j < population_size * 0.1; j++) {
            int subworstIndex = sortfitness[j].second;
            BinaryTree variation1 = generateExpressionTree2();
            for (int i = 0; i < population_size; i++) {
                At.clear();
                vector<float>resvarCal(200);
                vector<vector<float>> resvarData = FinData(variation1, data);
                for (int j = 0; j < 200; j++) {
                    vector<float> Data = resvarData[j];
                    revertExpressionTree(variation1, Data);
                    resvarCal[j] = variation1.evaluate();
                }
                revertAttributeValues(variation1, At);
                float varfitness = Cost(data, resvarCal);
                //population and fitness is renewed with new chromosomes
                population[subworstIndex] = variation1;
                fitness[subworstIndex] = varfitness;
            }
        }

    }

    // Find the best chromosome in the final population
    int bestIndex = distance(fitness.begin(), min_element(fitness.begin(), fitness.end()));
    BinaryTree bestIndividual = population[bestIndex];

    best_population.first = fitness[bestIndex];
    best_population.second = bestIndividual;
    return best_population; // The best combination of operators

}

// Expression Binary Tree Test program
int main()
{
    string filename = "dataset.csv";
    vector<vector<float>>data = ReadDataFromCSV(filename, 200, 7);
    pair<float, BinaryTree> bestIndividual;
    vector<pair<float, BinaryTree>> bestpopulation;
    //memo.clear();

    int population_size = 50; // Population size
    int generation_size = 10; // Number of generations

    for (int i = 0; i < 3; i++) {
        bestIndividual = geneticAlgorithm(data, population_size, generation_size);
        bestpopulation.push_back(bestIndividual);
    }

    sort(bestpopulation.begin(), bestpopulation.end(), compare);

    float best_fitness = bestpopulation[0].first;
    BinaryTree best_tree = bestpopulation[0].second;

    At.clear();
    cout << "Best fitness: " << best_fitness << endl;
    best_tree.inorder();
    cout << endl;
    // revertAttributeValues(best_tree, At);
    cout << "Best individual: ";
    best_tree.inorder();
    cout << endl;
    best_tree.inorder2(best_tree.getRoot());
    cout << endl;
    best_tree.post(best_tree.getRoot());

    return 0;
}