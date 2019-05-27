import XCTest
@testable import Denn

final class DennTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(Denn().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
