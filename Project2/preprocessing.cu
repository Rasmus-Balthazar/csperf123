#include "preprocessing.cuh"

__host__ int count_lines_in_file(const char *file_path) {
    std::ifstream inFile(file_path);  
    int line_count = std::count(std::istreambuf_iterator<char>(inFile), 
             std::istreambuf_iterator<char>(), '\n');
    return line_count;
}

__host__ char* read_file(const char* file_path) {
    std::ifstream t(file_path);
    std::string str;
    int file_len;

    t.seekg(0, std::ios::end);   
    file_len = t.tellg();
    str.reserve(file_len);
    t.seekg(0, std::ios::beg);

    str.assign((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());

    char *out_str = (char*)calloc(file_len, sizeof(char));
    memcpy(out_str, str.c_str(), file_len*sizeof(char));

    return out_str;
}

__host__ PatternsInformation process_patterns(const char *file_path) {
        std::string working_pattern;

        int num_lines = count_lines_in_file(file_path)-1;
        Pattern *patterns = (Pattern*)calloc(num_lines, sizeof(Pattern));

        int file_chars = 0;
        std::ifstream FileLenCounting(file_path);
        while (std::getline(FileLenCounting, working_pattern)) {
            file_chars += working_pattern.length()+1;
        }
        FileLenCounting.close(); 

        char* pattern_collection = (char*)calloc(file_chars, sizeof(char));
        int working_pattern_id = 0;
        int working_offset = 0;

        std::ifstream RegexFile(file_path);
        while (std::getline(RegexFile, working_pattern)) {
                patterns[working_pattern_id].pattern_len = working_pattern.length();
                patterns[working_pattern_id].pattern_text_offset = working_offset;
                memcpy(pattern_collection+working_offset, working_pattern.c_str(), (working_pattern.length()+1)*sizeof(char));
                working_offset += patterns[working_pattern_id].pattern_len+1;
                working_pattern_id++;
        }
        
        int format_len = patterns[num_lines-1].pattern_text_offset+patterns[num_lines-1].pattern_len;
        RegexFile.close(); 
        PatternsInformation info = {
            pattern_collection,
            patterns[num_lines-1].pattern_text_offset+patterns[num_lines-1].pattern_len,
            patterns,
            num_lines
        };

        return info;
}


__host__ int tokenize_helper(PatternsInformation p, int pos, Token* t) {
    switch (p.formatted_patterns[pos+1])
    {
    case '*':
        t->min_count=0u;
        t->max_count=-1u;
        return 1;
    case '+':
        t->min_count=1u;
        t->max_count=-1u;
        return 1;
    case '?':
        t->min_count=1u;
        t->max_count=-1u;
        return 1;
    default:
        t->min_count=1u;
        t->max_count=1u;
        return 0;
    }
}

__host__ RegEx* tokenize_regex(PatternsInformation p) {
    RegEx* regexes = (RegEx*)calloc(p.num_patterns, sizeof(RegEx));
    int token_offset = 0;

    for (int i = 0; i < p.num_patterns; i++)
    {
        Pattern* pattern = p.patterns+i;
        regexes[i].token_offset = token_offset;
        int token_count = 0;

        regexes[i].tokens = (Token*)calloc(pattern->pattern_len, sizeof(Token));
        for (int pattern_pos = pattern->pattern_text_offset; pattern_pos < pattern->pattern_len+pattern->pattern_text_offset; pattern_pos++)
        {
            switch (p.formatted_patterns[pattern_pos])
            {
            case '.': // wildcard token
                regexes[i].tokens[token_count].mode=1;
                regexes[i].tokens[token_count].to_match = '.';
                pattern_pos += tokenize_helper(p, pattern_pos, regexes[i].tokens+token_count);
                break;
            
            default: // literal token
                regexes[i].tokens[token_count].mode=0;
                regexes[i].tokens[token_count].to_match = p.formatted_patterns[pattern_pos];
                pattern_pos += tokenize_helper(p, pattern_pos, regexes[i].tokens+token_count);
                break;
            }
            token_count++;
        }
        regexes[i].token_count = token_count;
        token_offset += token_count;
        
    }
    return regexes;
}