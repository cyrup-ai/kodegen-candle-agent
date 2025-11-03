// Integration tests for domain operations

mod domain {
    mod chat {
        mod test_loop;
        mod message {
            mod test_message_processing;
            mod test_mod;
        }
        mod test_orchestration;
        mod templates {
            mod parser {
                mod test_mod;
            }
        }
    }
    mod model {
        mod test_error;
    }
    mod util {
        mod test_json_util;
    }
}
